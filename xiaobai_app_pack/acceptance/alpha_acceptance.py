from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from xiaobai_app_pack.launchers.official_app_launcher import (
    DEFAULT_APP_BASE,
    DEFAULT_PROFILE,
    DEFAULT_ROBOT_BASE,
    default_log_dir,
    request_json,
    result_to_dict,
    run_output_check,
    safe_request_json,
    sidecar_url,
    wait_for_official_app,
)
from xiaobai_app_pack.sidecars.memory_candidate_client import (
    MemoryCandidateClientError,
    create_candidate,
    delete_candidate,
)


DEFAULT_REQUIRED_TOOLS = {"move_head", "memory_candidate", "forget"}
DEFAULT_DISABLED_TOOLS = {"remember", "camera"}
NEXT_STEPS = {
    "official_app": [
        "Start Xiaobai with `scripts/start_xiaobai_official_app.sh`.",
        "Check `~/.local/state/xiaobai_app_pack/official_app.log` for backend connection errors.",
    ],
    "profile": [
        "Confirm REACHY_MINI_CUSTOM_PROFILE=xiaobai_app_pack_r1.",
        "Confirm REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY points to the App Pack profiles root.",
    ],
    "memory": [
        "Confirm Memory Candidate sidecar is running on port 8788.",
        "Run `python -m xiaobai_app_pack.sidecars.memory_candidate_cli candidates`.",
    ],
    "logs": [
        "Run the manual voice checks beside the robot, then rerun this acceptance command.",
        "If the log has no `role=user_partial`, check microphone routing in the official app.",
    ],
    "action_microphone": [
        "Stand beside Xiaobai and say exactly: `小白，请点一下头再说收到。`",
        "Rerun this acceptance command immediately after Xiaobai responds or stays silent.",
        "If no Chinese action phrase appears in the log, check microphone routing in the official app.",
    ],
    "action_llm": [
        "The action request reached the conversation log, but `move_head` was not called.",
        "Tighten the Xiaobai profile instruction for nod/head-move requests before changing runtime code.",
    ],
    "action_robot": [
        "`move_head` was called, but the tool or robot movement layer reported a failure.",
        "Run the R1-D output health check and inspect Reachy daemon/motor connection before retrying.",
    ],
    "output": [
        "Check Reachy daemon/media/volume/motor mode.",
        "Run the R1-D launcher health check before retrying conversation.",
    ],
}

ACTION_REQUEST_MARKERS = (
    "点头",
    "点一下头",
    "点点头",
    "动脑袋",
    "动动脑袋",
    "摇头",
    "抬头",
    "低头",
    "转头",
    "nod",
    "move your head",
    "move head",
)


@dataclass
class AcceptanceItem:
    name: str
    status: str
    evidence: dict[str, Any]
    next_steps: list[str]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def item(name: str, status: str, evidence: dict[str, Any] | None = None, next_steps: list[str] | None = None) -> dict:
    return {
        "name": name,
        "status": status,
        "evidence": evidence or {},
        "next_steps": next_steps or [],
    }


def launcher_item(result: Any) -> dict[str, Any]:
    data = result_to_dict(result)
    return item(
        str(data.get("name", "check")),
        "ok" if data.get("ok") is True else "fail",
        dict(data.get("details", {})),
        list(data.get("next_steps", [])),
    )


def load_profile(app_base: str, profile: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    query = urllib.parse.urlencode({"name": profile})
    data, error = safe_request_json(f"{app_base.rstrip('/')}/personalities/load?{query}", timeout=timeout)
    return (data if isinstance(data, dict) else None), error


def check_profile_tools(app_base: str, profile: str, timeout: float) -> dict[str, Any]:
    profile_data, error = load_profile(app_base, profile, timeout)
    enabled = set()
    if isinstance(profile_data, dict):
        raw_enabled = profile_data.get("enabled_tools", [])
        if isinstance(raw_enabled, list):
            enabled = {str(tool) for tool in raw_enabled}
    missing = sorted(DEFAULT_REQUIRED_TOOLS - enabled)
    unexpectedly_enabled = sorted(DEFAULT_DISABLED_TOOLS & enabled)
    status = "ok" if not error and not missing and not unexpectedly_enabled else "fail"
    return item(
        "profile_tools",
        status,
        {
            "profile": profile,
            "error": error,
            "enabled_tools": sorted(enabled),
            "missing_required": missing,
            "unexpectedly_enabled": unexpectedly_enabled,
        },
        [] if status == "ok" else NEXT_STEPS["profile"],
    )


def run_cli(args: Sequence[str], sidecar_base: str, timeout: float) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["XIAOBAI_MEMORY_SIDECAR_URL"] = sidecar_base
    env["XIAOBAI_MEMORY_SIDECAR_TIMEOUT"] = str(timeout)
    return subprocess.run(
        [sys.executable, "-m", "xiaobai_app_pack.sidecars.memory_candidate_cli", *args],
        capture_output=True,
        text=True,
        timeout=max(3.0, timeout + 2.0),
        env=env,
        check=False,
    )


def check_memory_candidate_flow(sidecar_base: str, timeout: float) -> dict[str, Any]:
    os.environ["XIAOBAI_MEMORY_SIDECAR_URL"] = sidecar_base
    os.environ["XIAOBAI_MEMORY_SIDECAR_TIMEOUT"] = str(timeout)
    fact = f"R1-E alpha acceptance marker {int(time.time())}"
    candidate_id = None
    evidence: dict[str, Any] = {"fact": fact}
    try:
        created = create_candidate(
            {
                "fact": fact,
                "subject": "xiaobai_acceptance",
                "confidence": "high",
                "sensitivity": "normal",
                "requested_by": "acceptance_runner",
            }
        )
        candidate = created.get("candidate", {})
        candidate_id = candidate.get("id")
        evidence["candidate_id"] = candidate_id
        evidence["candidate_status"] = candidate.get("status")

        candidates = run_cli(["candidates"], sidecar_base, timeout)
        evidence["cli_candidates_returncode"] = candidates.returncode
        evidence["cli_candidates_stdout"] = candidates.stdout[-500:]

        approved = run_cli(["approve", str(candidate_id)], sidecar_base, timeout)
        evidence["cli_approve_returncode"] = approved.returncode
        evidence["cli_approve_stdout"] = approved.stdout[-500:]

        forgotten = run_cli(["forget", fact], sidecar_base, timeout)
        evidence["cli_forget_returncode"] = forgotten.returncode
        evidence["cli_forget_stdout"] = forgotten.stdout[-500:]

        if candidate_id:
            deleted = delete_candidate(str(candidate_id))
            evidence["cleanup_candidate_status"] = deleted.get("candidate", {}).get("status")

        ok = (
            candidate_id
            and candidate.get("status") == "candidate"
            and candidates.returncode == 0
            and approved.returncode == 0
            and forgotten.returncode == 0
        )
        return item("memory_candidate_cli_flow", "ok" if ok else "fail", evidence, [] if ok else NEXT_STEPS["memory"])
    except (MemoryCandidateClientError, subprocess.SubprocessError, OSError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        if candidate_id:
            try:
                delete_candidate(str(candidate_id))
            except Exception:
                pass
        return item("memory_candidate_cli_flow", "fail", evidence, NEXT_STEPS["memory"])


def read_log_tail(log_file: Path, max_bytes: int = 200_000) -> str:
    if not log_file.exists():
        return ""
    size = log_file.stat().st_size
    with log_file.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace")


def scan_interaction_logs(log_file: Path) -> dict[str, Any]:
    text = read_log_tail(log_file)
    user_lines = [
        line
        for line in text.splitlines()
        if "role=user_partial content=" in line or "role=user content=" in line
    ][-12:]
    action_text_seen = any(marker.lower() in line.lower() for marker in ACTION_REQUEST_MARKERS for line in user_lines)
    move_head_tool_seen = "Tool call: move_head" in text or "tool_name='move_head'" in text
    move_head_tool_failed = (
        "move_head failed" in text
        or ("Tool 'move_head' (id=" in text and "failed" in text)
    )
    robot_motion_error_seen = "Failed to set robot target" in text or "Lost connection with the server" in text

    if move_head_tool_seen and move_head_tool_failed:
        action_diagnosis = "tool_called_robot_motion_failed"
        action_next_steps = NEXT_STEPS["action_robot"]
        action_status = "fail"
    elif move_head_tool_seen:
        action_diagnosis = "passed_move_head_tool_seen"
        action_next_steps = []
        action_status = "ok"
    elif action_text_seen:
        action_diagnosis = "llm_did_not_choose_move_head"
        action_next_steps = NEXT_STEPS["action_llm"]
        action_status = "fail"
    elif user_lines:
        action_diagnosis = "microphone_did_not_capture_action_request"
        action_next_steps = NEXT_STEPS["action_microphone"]
        action_status = "manual_pending"
    else:
        action_diagnosis = "no_user_voice_seen"
        action_next_steps = NEXT_STEPS["action_microphone"]
        action_status = "manual_pending"

    evidence = {
        "log_file": str(log_file),
        "has_log": bool(text),
        "assistant_reply_seen": "role=assistant content=" in text,
        "user_voice_seen": "role=user_partial content=" in text or "role=user content=" in text,
        "recent_user_lines": user_lines,
        "action_request_text_seen": action_text_seen,
        "action_request_diagnosis": action_diagnosis,
        "action_request_status": action_status,
        "move_head_tool_seen": move_head_tool_seen,
        "move_head_tool_failed": move_head_tool_failed,
        "robot_motion_error_seen": robot_motion_error_seen,
        "memory_candidate_tool_seen": "Tool call: memory_candidate" in text or "tool_name='memory_candidate'" in text,
        "xiaobai_profile_seen": "profile='xiaobai_app_pack_r1'" in text
        or "Loading tools for profile: xiaobai_app_pack_r1" in text,
    }
    manual_pending = []
    if not evidence["user_voice_seen"] or not evidence["assistant_reply_seen"]:
        manual_pending.append("short_qa")
    if action_status == "manual_pending":
        manual_pending.append("action_request")
    # Memory candidate is already verified through direct sidecar+CLI flow, so log evidence is optional.
    if action_status == "fail":
        status = "fail"
    elif manual_pending:
        status = "manual_pending"
    else:
        status = "ok"
    next_steps = []
    if status == "manual_pending":
        next_steps = NEXT_STEPS["logs"] + action_next_steps
    elif status == "fail":
        next_steps = action_next_steps
    return item(
        "interaction_log_evidence",
        status,
        {"manual_pending": manual_pending, **evidence},
        next_steps,
    )


def manual_script() -> str:
    return "\n".join(
        [
            "R1-E 家庭 alpha 真人验收话术：",
            "1. 短问答：对小白说：小白你好，请用一句话说你准备好了。",
            "   通过标准：小白在 3-6 秒内用中文回应，声音清楚，不连续重复。",
            "2. 动作请求：对小白说：小白，请点一下头再说收到。",
            "   通过标准：动作和语音都出现，动作不明显早于语音太久。",
            "3. 记忆候选：对小白说：小白，帮我记一下，陶陶喜欢恐龙英语单词。",
            "   通过标准：小白说明放入待确认记忆，不说已经永久记住。",
            "4. 验收后运行：python -m xiaobai_app_pack.acceptance.alpha_acceptance run",
            "   通过标准：报告里短问答/动作日志不再是 manual_pending。",
        ]
    )


def run_acceptance(args: argparse.Namespace) -> int:
    sidecar_base = args.sidecar_url or sidecar_url(args.sidecar_host, args.sidecar_port)
    results: list[dict[str, Any]] = []

    official = wait_for_official_app(args.app_base, timeout_seconds=args.ready_timeout)
    results.append(launcher_item(official))
    if official.ok:
        results.append(check_profile_tools(args.app_base, args.profile, args.timeout))
    else:
        results.append(item("profile_tools", "blocked", {"reason": "official_app_not_ready"}, NEXT_STEPS["official_app"]))

    if args.skip_output_check:
        results.append(item("reachy_output", "skipped", {"reason": "skip_output_check"}, []))
    else:
        results.append(launcher_item(run_output_check(args.robot_base, timeout=args.output_timeout)))

    results.append(check_memory_candidate_flow(sidecar_base, args.timeout))
    results.append(scan_interaction_logs(Path(args.log_file).expanduser()))

    report = {
        "name": "xiaobai_app_pack_r1e_alpha_acceptance",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "app_base": args.app_base,
        "sidecar_url": sidecar_base,
        "manual_script": manual_script().splitlines(),
        "results": results,
    }
    statuses = [str(result.get("status") if "status" in result else ("ok" if result.get("ok") else "fail")) for result in results]
    report["status"] = "pass" if all(status in {"ok", "skipped"} for status in statuses) else "needs_manual_or_fix"
    if args.report_file:
        report_file = Path(args.report_file).expanduser()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("xiaobai-alpha-acceptance")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--app-base", default=DEFAULT_APP_BASE)
    run.add_argument("--robot-base", default=DEFAULT_ROBOT_BASE)
    run.add_argument("--profile", default=DEFAULT_PROFILE)
    run.add_argument("--sidecar-host", default="127.0.0.1")
    run.add_argument("--sidecar-port", type=int, default=8788)
    run.add_argument("--sidecar-url", default=None)
    run.add_argument("--ready-timeout", type=float, default=5.0)
    run.add_argument("--timeout", type=float, default=3.0)
    run.add_argument("--output-timeout", type=float, default=6.0)
    run.add_argument("--skip-output-check", action="store_true")
    run.add_argument("--log-file", default=str(default_log_dir() / "official_app.log"))
    run.add_argument("--report-file", default=str(default_log_dir() / "r1e_alpha_acceptance_latest.json"))

    sub.add_parser("manual-script")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manual-script":
        print(manual_script())
        return 0
    if args.command == "run":
        return run_acceptance(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
