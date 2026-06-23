from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import ghoshell_moss_contrib.asr.async_states as async_states
from ghoshell_moss_contrib.asr.async_states import AsyncPdtListeningState


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now


class _Logger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class _Callback:
    def __init__(self) -> None:
        self.recognitions = []
        self.saved_batches = []

    async def on_recognition(self, result) -> None:
        self.recognitions.append(result)

    async def on_error(self, error: str) -> None:
        raise AssertionError(error)

    async def save_batch(self, rec, audio) -> None:
        self.saved_batches.append((rec, audio))


class _Batch:
    def __init__(self) -> None:
        self.buffered = []
        self.commits = 0

    async def buffer(self, audio) -> None:
        self.buffered.append(audio)

    async def commit(self) -> None:
        self.commits += 1

    async def wait_until_done(self) -> None:
        return None

    async def is_done(self) -> bool:
        return False

    async def get_buffer(self):
        if not self.buffered:
            return np.array([], dtype=np.int16)
        return np.concatenate(self.buffered)


class _CommitOnQuietVad:
    _speech_threshold = 600.0
    _silence_threshold = 400.0
    _silence_hold_time = 0.65
    _speech_detected = False
    _silence_start = None

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._calls = 0

    def __call__(self, audio, vad_time=None) -> bool:
        self._calls += 1
        if self._calls >= 2:
            self._clock.now += 0.3
            return True
        return False


@pytest.mark.asyncio
async def test_energy_vad_defers_empty_commit_for_local_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0.2")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_INPUT_GATE_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_AUDIO_IDLE_COMMIT_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_SPEECH_NO_TEXT_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_PRESPEECH_BATCH_MAX_SECONDS", "10")

    clock = _Clock()
    monkeypatch.setattr(async_states.time, "time", clock.time)

    callback = _Callback()
    batch = _Batch()
    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=callback,
        logger=_Logger(),
        vad=_CommitOnQuietVad(clock),
    )
    state._current_batch = batch
    state._batch_started_at = clock.time()

    fallback_calls = []

    async def _try_local_fallback(*, reason: str, max_rms: float, last_loud_age: float) -> str:
        fallback_calls.append((reason, max_rms, last_loud_age))
        return "小白你好"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert batch.commits == 0
    assert fallback_calls
    assert callback.recognitions[-1].text == "小白你好"
    assert callback.recognitions[-1].commit_reason == "local_whisper_quiet"
