import logging

import numpy as np
import pytest

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


@pytest.mark.asyncio
async def test_upload_player_finish_stream_invalidates_late_worker_frames(monkeypatch):
    player = ReachyMiniUploadAudioPlayer(
        _DummyMini(),
        logger=logging.getLogger("test_upload_player_finish_stream_invalidates_late_worker_frames"),
    )
    old_generation = player._generation
    uploads: list[str] = []
    monkeypatch.setattr(player, "_http_upload_and_play", lambda **_kwargs: uploads.append("upload"))
    monkeypatch.setattr(player, "_ensure_motors_ready", lambda **_kwargs: None)
    monkeypatch.setattr(player, "_log_output_health", lambda **_kwargs: None)

    await player.finish_stream()
    player._upload_and_play([np.zeros(2400, dtype=np.int16)], old_generation)

    assert uploads == []


def test_upload_player_drops_response_that_is_too_old_for_user_turn(monkeypatch):
    player = ReachyMiniUploadAudioPlayer(
        _DummyMini(),
        logger=logging.getLogger("test_upload_player_drops_response_that_is_too_old_for_user_turn"),
    )
    uploads: list[str] = []
    now = 1000.0
    monkeypatch.setattr("time.time", lambda: now)
    monkeypatch.setattr(
        "ghoshell_moss_contrib.moss_in_reachy_mini.audio.upload_player.latest_user_turn_at",
        lambda: now - 8.0,
    )
    monkeypatch.setattr(player, "_http_upload_and_play", lambda **_kwargs: uploads.append("upload"))
    monkeypatch.setattr(player, "_ensure_motors_ready", lambda **_kwargs: None)
    monkeypatch.setattr(player, "_log_output_health", lambda **_kwargs: None)

    player.mark_pending()
    player._upload_and_play([np.zeros(2400, dtype=np.int16)], player._generation)

    assert uploads == []


def test_upload_player_drops_when_new_user_turn_arrives_after_pending(monkeypatch):
    player = ReachyMiniUploadAudioPlayer(
        _DummyMini(),
        logger=logging.getLogger("test_upload_player_drops_when_new_user_turn_arrives_after_pending"),
    )
    uploads: list[str] = []
    now = 1000.0
    latest_turn = {"value": now - 1.0}
    monkeypatch.setattr("time.time", lambda: now)
    monkeypatch.setattr(
        "ghoshell_moss_contrib.moss_in_reachy_mini.audio.upload_player.latest_user_turn_at",
        lambda: latest_turn["value"],
    )
    monkeypatch.setattr(player, "_http_upload_and_play", lambda **_kwargs: uploads.append("upload"))
    monkeypatch.setattr(player, "_ensure_motors_ready", lambda **_kwargs: None)
    monkeypatch.setattr(player, "_log_output_health", lambda **_kwargs: None)

    player.mark_pending()
    latest_turn["value"] = now + 1.0
    player._upload_and_play([np.zeros(2400, dtype=np.int16)], player._generation)

    assert uploads == []


def test_upload_player_keeps_non_voice_tts_when_last_voice_turn_is_old(monkeypatch):
    player = ReachyMiniUploadAudioPlayer(
        _DummyMini(),
        logger=logging.getLogger("test_upload_player_keeps_non_voice_tts_when_last_voice_turn_is_old"),
    )
    uploads: list[str] = []
    now = 1000.0
    monkeypatch.setattr("time.time", lambda: now)
    monkeypatch.setattr(
        "ghoshell_moss_contrib.moss_in_reachy_mini.audio.upload_player.latest_user_turn_at",
        lambda: now - 120.0,
    )
    monkeypatch.setattr(player, "_http_upload_and_play", lambda **_kwargs: uploads.append("upload"))
    monkeypatch.setattr(player, "_ensure_motors_ready", lambda **_kwargs: None)
    monkeypatch.setattr(player, "_log_output_health", lambda **_kwargs: None)

    player.mark_pending()
    player._upload_and_play([np.zeros(2400, dtype=np.int16)], player._generation)

    assert uploads == ["upload"]
