import wave

import pytest

from ghoshell_moss_contrib.moss_in_reachy_mini.audio import sound_cues
from ghoshell_moss_contrib.moss_in_reachy_mini.audio.sound_cues import (
    SoundCuePlayer,
    make_cue_wav,
)


class FakeCueTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, float]] = []

    def play_wav(self, *, name: str, wav_bytes: bytes, timeout: float) -> dict[str, object]:
        self.calls.append((name, wav_bytes, timeout))
        return {"played": {"status": "ok"}}


def test_make_cue_wav_produces_valid_wav() -> None:
    wav_bytes, duration = make_cue_wav("thinking")

    assert duration > 0.05
    with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() > 0


def test_sound_cue_disabled_does_not_call_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeCueTransport()
    monkeypatch.setattr(sound_cues, "mark_speaking_for", lambda *_args, **_kwargs: None)
    player = SoundCuePlayer(transport, enabled=False)

    result = player.play("thinking")

    assert not result.scheduled
    assert result.reason == "disabled"
    assert transport.calls == []


def test_sound_cue_schedules_transport_and_marks_output_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeCueTransport()
    marked: list[tuple[float, float]] = []
    monkeypatch.setattr(sound_cues, "is_speaking", lambda **_kwargs: False)
    monkeypatch.setattr(sound_cues, "is_speech_pending", lambda: False)
    monkeypatch.setattr(
        sound_cues,
        "mark_speaking_for",
        lambda duration, *, tail=0.0: marked.append((duration, tail)),
    )
    player = SoundCuePlayer(transport, enabled=True, cooldown_seconds=0.0, timeout_seconds=1.2)

    result = player.play("start-listening")

    assert result.scheduled
    assert result.kind == "listening"
    assert player.wait_idle(timeout=1.0)
    assert len(transport.calls) == 1
    name, wav_bytes, timeout = transport.calls[0]
    assert name.startswith("moss_cue_listening_")
    assert len(wav_bytes) > 100
    assert timeout == 1.2
    assert marked and marked[0][0] == result.duration_seconds


def test_sound_cue_suppresses_when_robot_output_active(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeCueTransport()
    monkeypatch.setattr(sound_cues, "is_speaking", lambda **_kwargs: True)
    monkeypatch.setattr(sound_cues, "is_speech_pending", lambda: False)
    player = SoundCuePlayer(transport, enabled=True)

    result = player.play("thinking")

    assert not result.scheduled
    assert result.reason == "robot_output_active"
    assert transport.calls == []


def test_sound_cue_cooldown_drops_repeated_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeCueTransport()
    monkeypatch.setattr(sound_cues, "is_speaking", lambda **_kwargs: False)
    monkeypatch.setattr(sound_cues, "is_speech_pending", lambda: False)
    monkeypatch.setattr(sound_cues, "mark_speaking_for", lambda *_args, **_kwargs: None)
    player = SoundCuePlayer(transport, enabled=True, cooldown_seconds=10.0)

    first = player.play("error")
    second = player.play("error")

    assert first.scheduled
    assert not second.scheduled
    assert second.reason == "cooldown"
    assert player.wait_idle(timeout=1.0)
    assert len(transport.calls) == 1
