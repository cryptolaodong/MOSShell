from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_PROFILE = "xiaobai_app_pack_r1"
DEFAULT_SIDECAR_HOST = "127.0.0.1"
DEFAULT_SIDECAR_PORT = 8788
DEFAULT_APP_BASE = "http://127.0.0.1:7860"
DEFAULT_ROBOT_BASE = "http://192.168.31.222:8000"
DEFAULT_OLD_MOSS_SCREEN = "moss-ghost"
DEFAULT_TRANSCRIPTION_LANGUAGE = "zh"

NEXT_STEPS = {
    "old_moss_running": [
        "Stop the old MOSS runtime, or rerun with the default --stop-old-moss behavior.",
        "Check `screen -ls` and confirm only one robot runtime is controlling media/action.",
    ],
    "sidecar_unavailable": [
        "Start the memory sidecar with `python -m xiaobai_app_pack.sidecars.memory_candidate_sidecar`.",
        "Check XIAOBAI_MEMORY_SIDECAR_URL and port 8788.",
    ],
    "app_not_ready": [
        "Open the app log printed by the launcher.",
        "Check the official app `/status` endpoint and backend connection.",
    ],
    "output_check_failed": [
        "Run the output self-check again after confirming Reachy daemon is reachable.",
        "Check daemon, media acquire, volume, and motor mode before testing conversation.",
    ],
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: dict[str, Any]
    next_steps: list[str]


class LauncherError(RuntimeError):
    def __init__(self, result: CheckResult) -> None:
        super().__init__(result.name)
        self.result = result


def app_pack_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return app_pack_root().parent


def default_log_dir() -> Path:
    return Path(os.getenv("XIAOBAI_APP_PACK_LOG_DIR", "~/.local/state/xiaobai_app_pack")).expanduser()


def sidecar_url(host: str = DEFAULT_SIDECAR_HOST, port: int = DEFAULT_SIDECAR_PORT) -> str:
    return f"http://{host}:{port}"


def build_official_env(
    *,
    base_env: Mapping[str, str] | None = None,
    profile: str = DEFAULT_PROFILE,
    sidecar_base_url: str | None = None,
    app_timeout_minutes: str = "0",
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["REACHY_MINI_CUSTOM_PROFILE"] = profile
    env["REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY"] = str(app_pack_root() / "profiles" / "official_app")
    env["XIAOBAI_MEMORY_SIDECAR_URL"] = sidecar_base_url or sidecar_url()
    env.setdefault("REALTIME_TRANSCRIPTION_LANGUAGE", DEFAULT_TRANSCRIPTION_LANGUAGE)
    env.setdefault("REACHY_MINI_APP_TIMEOUT_MINUTES", app_timeout_minutes)
    return env


def build_sidecar_command(*, host: str = DEFAULT_SIDECAR_HOST, port: int = DEFAULT_SIDECAR_PORT) -> list[str]:
    return [
        sys.executable,
        "-m",
        "xiaobai_app_pack.sidecars.memory_candidate_sidecar",
        "--host",
        host,
        "--port",
        str(port),
    ]


def build_official_app_command(*, debug: bool = False, robot_name: str | None = None) -> list[str]:
    command = ["uv", "run", "reachy-mini-conversation-app", "--no-camera", "--ui"]
    if debug:
        command.append("--debug")
    if robot_name:
        command.extend(["--robot-name", robot_name])
    return command


def request_json(url: str, *, timeout: float = 3.0, method: str = "GET") -> Any:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def post_json(base_url: str, path: str, *, timeout: float = 5.0) -> Any:
    return request_json(f"{base_url.rstrip('/')}{path}", timeout=timeout, method="POST")


def safe_request_json(url: str, *, timeout: float = 3.0, method: str = "GET") -> tuple[Any | None, str | None]:
    try:
        return request_json(url, timeout=timeout, method=method), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def check_sidecar(base_url: str, *, timeout: float = 2.0) -> CheckResult:
    data, error = safe_request_json(f"{base_url.rstrip('/')}/health", timeout=timeout)
    ok = isinstance(data, dict) and data.get("ok") is True
    return CheckResult(
        name="memory_sidecar",
        ok=ok,
        details={"url": base_url, "error": error, "response": data},
        next_steps=[] if ok else NEXT_STEPS["sidecar_unavailable"],
    )


def wait_for_sidecar(base_url: str, *, timeout_seconds: float = 8.0) -> CheckResult:
    deadline = time.time() + timeout_seconds
    last = check_sidecar(base_url)
    while time.time() < deadline:
        last = check_sidecar(base_url)
        if last.ok:
            return last
        time.sleep(0.25)
    return last


def wait_for_official_app(app_base: str, *, timeout_seconds: float = 60.0) -> CheckResult:
    deadline = time.time() + timeout_seconds
    details: dict[str, Any] = {"app_base": app_base}
    while time.time() < deadline:
        ready, ready_error = safe_request_json(f"{app_base.rstrip('/')}/ready", timeout=3.0)
        status, status_error = safe_request_json(f"{app_base.rstrip('/')}/status", timeout=3.0)
        details = {
            "app_base": app_base,
            "ready": ready,
            "ready_error": ready_error,
            "status": status,
            "status_error": status_error,
        }
        ready_ok = isinstance(ready, dict) and ready.get("ready") is True
        status_ok = isinstance(status, dict) and status.get("backend_connected") is True
        if ready_ok and status_ok:
            return CheckResult("official_app_ready", True, details, [])
        time.sleep(1.0)
    return CheckResult("official_app_ready", False, details, NEXT_STEPS["app_not_ready"])


def run_output_check(robot_base: str, *, timeout: float = 6.0) -> CheckResult:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name, path in {
        "daemon": "/api/daemon/status",
        "media": "/api/media/status",
        "volume": "/api/volume/current",
        "motors": "/api/motors/status",
    }.items():
        data, error = safe_request_json(f"{robot_base.rstrip('/')}{path}", timeout=timeout)
        checks[name] = {"response": data, "error": error}
        if error:
            failures.append(name)

    daemon = checks.get("daemon", {}).get("response")
    if isinstance(daemon, dict) and daemon.get("state") != "running":
        failures.append("daemon_not_running")

    for name, path in {
        "stop_sound": "/api/media/stop_sound",
        "media_acquire": "/api/media/acquire",
        "motors_enable": "/api/motors/set_mode/enabled",
        "test_sound": "/api/volume/test-sound",
    }.items():
        data, error = safe_request_json(f"{robot_base.rstrip('/')}{path}", timeout=timeout, method="POST")
        checks[name] = {"response": data, "error": error}
        if error:
            failures.append(name)
    test_sound = checks.get("test_sound", {}).get("response")
    if isinstance(test_sound, dict) and test_sound.get("status") != "ok":
        failures.append("test_sound_status")

    return CheckResult(
        name="reachy_output",
        ok=not failures,
        details={"robot_base": robot_base, "checks": checks, "failures": failures},
        next_steps=[] if not failures else NEXT_STEPS["output_check_failed"],
    )


def stop_old_moss_runtime(
    *,
    screen_name: str = DEFAULT_OLD_MOSS_SCREEN,
    stop: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CheckResult:
    if shutil.which("screen") is None:
        return CheckResult("old_moss_runtime", True, {"screen_available": False}, [])

    listed = runner(["screen", "-ls"], capture_output=True, text=True, check=False)
    output = (listed.stdout or "") + (listed.stderr or "")
    is_running = screen_name in output
    details: dict[str, Any] = {"screen_name": screen_name, "running": is_running}
    if not is_running:
        return CheckResult("old_moss_runtime", True, details, [])
    if not stop:
        return CheckResult("old_moss_runtime", False, details, NEXT_STEPS["old_moss_running"])
    stopped = runner(["screen", "-S", screen_name, "-X", "quit"], capture_output=True, text=True, check=False)
    details["stopped_returncode"] = stopped.returncode
    return CheckResult("old_moss_runtime", stopped.returncode == 0, details, NEXT_STEPS["old_moss_running"])


def ensure_log_dir(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def start_process(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_file: Path) -> subprocess.Popen[Any]:
    handle = log_file.open("ab")
    return subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def result_to_dict(result: CheckResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "ok": result.ok,
        "details": result.details,
        "next_steps": result.next_steps,
    }


def print_result(result: CheckResult) -> None:
    print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2, sort_keys=True))


def start(args: argparse.Namespace) -> int:
    log_dir = ensure_log_dir(Path(args.log_dir).expanduser())
    official_root = Path(args.official_root).expanduser().resolve()
    sidecar_base = args.sidecar_url or sidecar_url(args.sidecar_host, args.sidecar_port)

    old_moss = stop_old_moss_runtime(screen_name=args.old_moss_screen, stop=not args.no_stop_old_moss)
    if not old_moss.ok:
        raise LauncherError(old_moss)

    env = build_official_env(
        profile=args.profile,
        sidecar_base_url=sidecar_base,
        app_timeout_minutes=str(args.app_timeout_minutes),
    )
    env["PYTHONPATH"] = str(repo_root()) + os.pathsep + env.get("PYTHONPATH", "")

    sidecar = check_sidecar(sidecar_base)
    sidecar_process = None
    if not sidecar.ok:
        sidecar_command = build_sidecar_command(host=args.sidecar_host, port=args.sidecar_port)
        if args.dry_run:
            print("would_start_sidecar=" + " ".join(sidecar_command))
            sidecar = CheckResult("memory_sidecar", True, {"dry_run": True, "url": sidecar_base}, [])
        else:
            sidecar_process = start_process(
                sidecar_command,
                cwd=repo_root(),
                env=env,
                log_file=log_dir / "memory_sidecar.log",
            )
            sidecar = wait_for_sidecar(sidecar_base)
    if not sidecar.ok:
        raise LauncherError(sidecar)

    app_command = build_official_app_command(debug=args.debug, robot_name=args.robot_name)
    if args.dry_run:
        print("would_start_official_app=" + " ".join(app_command))
        print(f"REACHY_MINI_CUSTOM_PROFILE={env['REACHY_MINI_CUSTOM_PROFILE']}")
        print(f"REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY={env['REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY']}")
        print(f"XIAOBAI_MEMORY_SIDECAR_URL={env['XIAOBAI_MEMORY_SIDECAR_URL']}")
        print(f"REALTIME_TRANSCRIPTION_LANGUAGE={env['REALTIME_TRANSCRIPTION_LANGUAGE']}")
        return 0

    app_process = start_process(app_command, cwd=official_root, env=env, log_file=log_dir / "official_app.log")
    ready = wait_for_official_app(args.app_base, timeout_seconds=args.ready_timeout)
    if not ready.ok:
        if app_process.poll() is not None:
            ready.details["app_process_returncode"] = app_process.returncode
        if sidecar_process is not None:
            ready.details["sidecar_pid"] = sidecar_process.pid
        raise LauncherError(ready)

    output = CheckResult("reachy_output", True, {"skipped": True}, [])
    if not args.skip_output_check:
        output = run_output_check(args.robot_base, timeout=args.output_timeout)
        if not output.ok:
            raise LauncherError(output)

    print_result(
        CheckResult(
            "xiaobai_official_app_started",
            True,
            {
                "app_pid": app_process.pid,
                "sidecar_pid": getattr(sidecar_process, "pid", None),
                "app_base": args.app_base,
                "sidecar_url": sidecar_base,
                "log_dir": str(log_dir),
                "old_moss": result_to_dict(old_moss),
                "ready": result_to_dict(ready),
                "output": result_to_dict(output),
            },
            [],
        )
    )
    return 0


def check(args: argparse.Namespace) -> int:
    results = [
        check_sidecar(args.sidecar_url or sidecar_url(args.sidecar_host, args.sidecar_port)),
        wait_for_official_app(args.app_base, timeout_seconds=args.ready_timeout),
    ]
    if not args.skip_output_check:
        results.append(run_output_check(args.robot_base, timeout=args.output_timeout))
    ok = all(result.ok for result in results)
    print(json.dumps([result_to_dict(result) for result in results], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


def print_env(args: argparse.Namespace) -> int:
    env = build_official_env(profile=args.profile, sidecar_base_url=args.sidecar_url)
    for key in (
        "REACHY_MINI_CUSTOM_PROFILE",
        "REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY",
        "XIAOBAI_MEMORY_SIDECAR_URL",
        "REALTIME_TRANSCRIPTION_LANGUAGE",
    ):
        print(f"export {key}={json.dumps(env[key], ensure_ascii=False)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("xiaobai-official-app-launcher")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "check", "env"):
        command = sub.add_parser(name)
        command.add_argument("--profile", default=DEFAULT_PROFILE)
        command.add_argument("--sidecar-host", default=DEFAULT_SIDECAR_HOST)
        command.add_argument("--sidecar-port", type=int, default=DEFAULT_SIDECAR_PORT)
        command.add_argument("--sidecar-url", default=None)
        command.add_argument("--app-base", default=DEFAULT_APP_BASE)
        command.add_argument("--robot-base", default=DEFAULT_ROBOT_BASE)
        command.add_argument("--ready-timeout", type=float, default=60.0)
        command.add_argument("--output-timeout", type=float, default=6.0)
        command.add_argument("--skip-output-check", action="store_true")
    start_parser = sub.choices["start"]
    start_parser.add_argument("--official-root", default=str(repo_root()))
    start_parser.add_argument("--log-dir", default=str(default_log_dir()))
    start_parser.add_argument("--debug", action="store_true")
    start_parser.add_argument("--dry-run", action="store_true")
    start_parser.add_argument("--robot-name", default=None)
    start_parser.add_argument("--app-timeout-minutes", default="0")
    start_parser.add_argument("--old-moss-screen", default=DEFAULT_OLD_MOSS_SCREEN)
    start_parser.add_argument("--no-stop-old-moss", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "start":
            return start(args)
        if args.command == "check":
            return check(args)
        if args.command == "env":
            return print_env(args)
    except LauncherError as exc:
        print_result(exc.result)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
