from __future__ import annotations

import subprocess
from types import SimpleNamespace

import xiaobai_app_pack.launchers.official_app_launcher as launcher


def test_build_official_env_points_to_external_profile(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "app_pack_root", lambda: launcher.Path("/repo/xiaobai_app_pack"))
    env = launcher.build_official_env(base_env={}, sidecar_base_url="http://127.0.0.1:8788")
    assert env["REACHY_MINI_CUSTOM_PROFILE"] == "xiaobai_app_pack_r1"
    assert env["REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY"] == "/repo/xiaobai_app_pack/profiles/official_app"
    assert env["XIAOBAI_MEMORY_SIDECAR_URL"] == "http://127.0.0.1:8788"
    assert env["REALTIME_TRANSCRIPTION_LANGUAGE"] == "zh"
    assert env["REACHY_MINI_APP_TIMEOUT_MINUTES"] == "0"


def test_build_official_env_preserves_explicit_transcription_language(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "app_pack_root", lambda: launcher.Path("/repo/xiaobai_app_pack"))
    env = launcher.build_official_env(
        base_env={"REALTIME_TRANSCRIPTION_LANGUAGE": "en"},
        sidecar_base_url="http://127.0.0.1:8788",
    )
    assert env["REALTIME_TRANSCRIPTION_LANGUAGE"] == "en"


def test_build_commands_are_official_app_and_sidecar_only() -> None:
    assert launcher.build_sidecar_command(host="127.0.0.1", port=8788)[1:4] == [
        "-m",
        "xiaobai_app_pack.sidecars.memory_candidate_sidecar",
        "--host",
    ]
    command = launcher.build_official_app_command(debug=True, robot_name="mini")
    assert command[:3] == ["uv", "run", "reachy-mini-conversation-app"]
    assert "--no-camera" in command
    assert "--ui" in command
    assert "--debug" in command
    assert command[-2:] == ["--robot-name", "mini"]


def test_stop_old_moss_runtime_stops_screen(monkeypatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/screen")
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        if command == ["screen", "-ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="123.moss-ghost\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = launcher.stop_old_moss_runtime(runner=fake_runner)
    assert result.ok is True
    assert ["screen", "-S", "moss-ghost", "-X", "quit"] in calls


def test_stop_old_moss_runtime_can_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/screen")

    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="123.moss-ghost\n", stderr="")

    result = launcher.stop_old_moss_runtime(stop=False, runner=fake_runner)
    assert result.ok is False
    assert result.name == "old_moss_runtime"
    assert result.next_steps


def test_output_check_failure_has_next_steps(monkeypatch) -> None:
    def fake_safe_request(url, *, timeout=3.0, method="GET"):
        return None, "timeout"

    monkeypatch.setattr(launcher, "safe_request_json", fake_safe_request)
    result = launcher.run_output_check("http://robot.local:8000")
    assert result.ok is False
    assert "daemon" in result.details["failures"]
    assert result.next_steps


def test_dry_run_prints_env_and_does_not_start_process(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(launcher, "check_sidecar", lambda base_url: launcher.CheckResult("memory_sidecar", True, {}, []))
    monkeypatch.setattr(launcher, "stop_old_moss_runtime", lambda **kwargs: launcher.CheckResult("old", True, {}, []))

    def fail_start(*args, **kwargs):
        raise AssertionError("dry-run must not start a process")

    monkeypatch.setattr(launcher, "start_process", fail_start)
    args = SimpleNamespace(
        log_dir=str(tmp_path),
        official_root=str(tmp_path),
        sidecar_url=None,
        sidecar_host="127.0.0.1",
        sidecar_port=8788,
        old_moss_screen="moss-ghost",
        no_stop_old_moss=False,
        profile="xiaobai_app_pack_r1",
        app_timeout_minutes="0",
        debug=False,
        robot_name=None,
        dry_run=True,
        app_base="http://127.0.0.1:7860",
        ready_timeout=1.0,
        skip_output_check=True,
        robot_base="http://robot.local:8000",
        output_timeout=1.0,
    )
    assert launcher.start(args) == 0
    out = capsys.readouterr().out
    assert "would_start_official_app=uv run reachy-mini-conversation-app --no-camera --ui" in out
    assert "REACHY_MINI_CUSTOM_PROFILE=xiaobai_app_pack_r1" in out
    assert "REALTIME_TRANSCRIPTION_LANGUAGE=zh" in out
