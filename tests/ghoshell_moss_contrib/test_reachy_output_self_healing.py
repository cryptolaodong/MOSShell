import logging

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.upload_player import (
    ReachyMiniUploadAudioPlayer,
)


class _DummyMini:
    host = "127.0.0.1"
    port = 8000


def test_upload_player_auto_enables_disabled_motors(monkeypatch):
    player = ReachyMiniUploadAudioPlayer(
        _DummyMini(),
        logger=logging.getLogger("test_upload_player_auto_enables_disabled_motors"),
    )
    calls: list[tuple[str, str]] = []
    status_calls = 0

    def fake_http_json(path, *, method="GET", data=None, timeout=None):
        nonlocal status_calls
        calls.append((method, path))
        if path == "/api/motors/status":
            status_calls += 1
            return {"mode": "disabled"} if status_calls == 1 else {"mode": "enabled"}
        if path == "/api/motors/set_mode/enabled":
            return {"status": "motors changed to MotorControlMode.Enabled mode"}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(player, "_http_json", fake_http_json)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    player._ensure_motors_ready(reason="test")

    assert ("POST", "/api/motors/set_mode/enabled") in calls
    assert calls[-1] == ("GET", "/api/motors/status")


def test_upload_player_respects_auto_enable_motors_env(monkeypatch):
    monkeypatch.setenv("MOSS_REACHY_AUTO_ENABLE_MOTORS", "0")
    player = ReachyMiniUploadAudioPlayer(
        _DummyMini(),
        logger=logging.getLogger("test_upload_player_respects_auto_enable_motors_env"),
    )
    calls: list[str] = []
    monkeypatch.setattr(player, "_http_json", lambda path, **_kwargs: calls.append(path))

    player._ensure_motors_ready(reason="test")

    assert calls == []
