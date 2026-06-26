from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import xiaobai_app_pack.acceptance.alpha_acceptance as acceptance
from xiaobai_app_pack.launchers.official_app_launcher import CheckResult


def test_manual_script_contains_core_family_alpha_steps() -> None:
    script = acceptance.manual_script()
    assert "短问答" in script
    assert "动作请求" in script
    assert "记忆候选" in script
    assert "待确认记忆" in script


def test_profile_tools_acceptance_detects_required_and_disabled_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        acceptance,
        "load_profile",
        lambda *_args, **_kwargs: (
            {"enabled_tools": ["move_head", "memory_candidate", "forget"]},
            None,
        ),
    )
    result = acceptance.check_profile_tools("http://app", "xiaobai_app_pack_r1", 1.0)
    assert result["status"] == "ok"
    assert result["evidence"]["missing_required"] == []
    assert result["evidence"]["unexpectedly_enabled"] == []


def test_profile_tools_acceptance_rejects_direct_remember(monkeypatch) -> None:
    monkeypatch.setattr(
        acceptance,
        "load_profile",
        lambda *_args, **_kwargs: (
            {"enabled_tools": ["move_head", "memory_candidate", "forget", "remember"]},
            None,
        ),
    )
    result = acceptance.check_profile_tools("http://app", "xiaobai_app_pack_r1", 1.0)
    assert result["status"] == "fail"
    assert result["evidence"]["unexpectedly_enabled"] == ["remember"]
    assert result["next_steps"]


def test_log_scan_marks_manual_pending_until_voice_and_action_seen(tmp_path: Path) -> None:
    log = tmp_path / "official_app.log"
    log.write_text("profile='xiaobai_app_pack_r1'\nrole=assistant content=你好\n", encoding="utf-8")
    result = acceptance.scan_interaction_logs(log)
    assert result["status"] == "manual_pending"
    assert "short_qa" in result["evidence"]["manual_pending"]
    assert "action_request" in result["evidence"]["manual_pending"]


def test_log_scan_passes_after_voice_and_move_head_seen(tmp_path: Path) -> None:
    log = tmp_path / "official_app.log"
    log.write_text(
        "\n".join(
            [
                "profile='xiaobai_app_pack_r1'",
                "role=user_partial content=小白你好",
                "role=assistant content=我准备好了",
                "Tool call: move_head direction=up",
            ]
        ),
        encoding="utf-8",
    )
    result = acceptance.scan_interaction_logs(log)
    assert result["status"] == "ok"
    assert result["evidence"]["manual_pending"] == []


def test_memory_candidate_flow_uses_cli_approve_and_forget(monkeypatch) -> None:
    cli_calls: list[list[str]] = []

    monkeypatch.setattr(
        acceptance,
        "create_candidate",
        lambda payload: {"candidate": {"id": "cand-1", "status": "candidate", "fact": payload["fact"]}},
    )
    monkeypatch.setattr(
        acceptance,
        "delete_candidate",
        lambda candidate_id: {"candidate": {"id": candidate_id, "status": "deleted"}},
    )

    def fake_cli(args, sidecar_base, timeout):
        cli_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(acceptance, "run_cli", fake_cli)
    result = acceptance.check_memory_candidate_flow("http://127.0.0.1:8788", 1.0)
    assert result["status"] == "ok"
    assert cli_calls[0] == ["candidates"]
    assert cli_calls[1] == ["approve", "cand-1"]
    assert cli_calls[2][0] == "forget"
    assert cli_calls[2][1].startswith("R1-E alpha acceptance marker")


def test_run_acceptance_reports_manual_pending(monkeypatch, tmp_path: Path, capsys) -> None:
    log = tmp_path / "official_app.log"
    log.write_text("profile='xiaobai_app_pack_r1'\nrole=assistant content=你好\n", encoding="utf-8")
    monkeypatch.setattr(
        acceptance,
        "wait_for_official_app",
        lambda *_args, **_kwargs: CheckResult("official_app_ready", True, {"ready": True}, []),
    )
    monkeypatch.setattr(
        acceptance,
        "check_profile_tools",
        lambda *_args, **_kwargs: acceptance.item("profile_tools", "ok", {}, []),
    )
    monkeypatch.setattr(
        acceptance,
        "check_memory_candidate_flow",
        lambda *_args, **_kwargs: acceptance.item("memory_candidate_cli_flow", "ok", {}, []),
    )
    args = SimpleNamespace(
        sidecar_url="http://127.0.0.1:8788",
        sidecar_host="127.0.0.1",
        sidecar_port=8788,
        app_base="http://127.0.0.1:7860",
        robot_base="http://robot.local:8000",
        profile="xiaobai_app_pack_r1",
        timeout=1.0,
        ready_timeout=1.0,
        output_timeout=1.0,
        skip_output_check=True,
        log_file=str(log),
        report_file=str(tmp_path / "report.json"),
    )
    assert acceptance.run_acceptance(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "needs_manual_or_fix"
    assert any(result["status"] == "manual_pending" for result in report["results"])
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["status"] == "needs_manual_or_fix"
