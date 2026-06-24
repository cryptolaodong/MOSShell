import asyncio
import logging
from collections.abc import AsyncIterable

import numpy as np
import pytest

from ghoshell_moss.contracts.speech import AudioFormat, StreamAudioPlayer, TTSBatch, TTSItem
from ghoshell_moss.core.speech.stream_tts_speech import TTSSpeechStream


class _FakeBatch(TTSBatch):
    def __init__(self, *, audio_seconds: float = 0.1, sample_rate: int = 24000):
        self._batch_id = "fake-batch"
        self._audio_seconds = audio_seconds
        self._sample_rate = sample_rate
        self._started = False
        self._committed = False
        self._closed = False

    def batch_id(self) -> str:
        return self._batch_id

    def with_callback(self, callback) -> None:
        pass

    def feed(self, text: str):
        pass

    def commit(self):
        self._committed = True

    async def start(self) -> None:
        self._started = True

    async def close(self) -> None:
        self._closed = True

    def is_committed(self) -> bool:
        return self._committed

    def is_closed(self) -> bool:
        return self._closed

    def is_started(self) -> bool:
        return self._started

    async def items(self) -> AsyncIterable[TTSItem]:
        audio = np.zeros(int(self._sample_rate * self._audio_seconds), dtype=np.int16)
        yield {
            "text": "收到",
            "audio": audio,
            "sample_rate": self._sample_rate,
            "audio_format": AudioFormat.PCM_S16LE.value,
            "channels": 1,
            "tone": "",
            "voice": {},
        }

    async def wait_done(self, timeout: float | None = None):
        return True


class _SlowDonePlayer(StreamAudioPlayer):
    audio_type = AudioFormat.PCM_S16LE
    channels = 1
    sample_rate = 24000

    def __init__(self, *, play_seconds: float):
        self.play_seconds = play_seconds
        self.clear_calls = 0
        self.finish_calls = 0
        self.started = False
        self.closed = False
        self.playing = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def clear(self) -> None:
        self.clear_calls += 1
        self.playing = False

    async def finish_stream(self) -> None:
        self.finish_calls += 1

    def add(self, chunk: np.ndarray, *, audio_type: AudioFormat, rate: int, channels: int = 1) -> float:
        self.playing = True
        return 0.0

    async def wait_play_done(self, timeout: float | None = None) -> bool:
        await asyncio.sleep(self.play_seconds)
        self.playing = False
        return True

    def is_playing(self) -> bool:
        return self.playing

    def is_closed(self) -> bool:
        return self.closed

    def on_play(self, callback) -> None:
        pass

    def on_play_done(self, callback) -> None:
        pass


@pytest.mark.asyncio
async def test_tts_stream_close_waits_for_active_playback(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_CLOSE_PLAYBACK_GRACE_SECONDS", "0.5")
    player = _SlowDonePlayer(play_seconds=0.05)
    stream = TTSSpeechStream(
        loop=asyncio.get_running_loop(),
        audio_format=AudioFormat.PCM_S16LE,
        channels=1,
        sample_rate=24000,
        player=player,
        tts_batch=_FakeBatch(),
        logger=logging.getLogger("test_tts_stream_close_waits_for_active_playback"),
    )

    await stream.start_synthesis()
    await stream.start_play()
    await asyncio.sleep(0)
    await stream.close()

    assert player.clear_calls == 1
    assert player.finish_calls >= 1


@pytest.mark.asyncio
async def test_tts_stream_close_hard_clears_after_playback_grace_timeout(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_CLOSE_PLAYBACK_GRACE_SECONDS", "0.01")
    player = _SlowDonePlayer(play_seconds=0.2)
    stream = TTSSpeechStream(
        loop=asyncio.get_running_loop(),
        audio_format=AudioFormat.PCM_S16LE,
        channels=1,
        sample_rate=24000,
        player=player,
        tts_batch=_FakeBatch(),
        logger=logging.getLogger("test_tts_stream_close_hard_clears_after_playback_grace_timeout"),
    )

    await stream.start_synthesis()
    await stream.start_play()
    await asyncio.sleep(0)
    await stream.close()

    assert player.clear_calls >= 2
