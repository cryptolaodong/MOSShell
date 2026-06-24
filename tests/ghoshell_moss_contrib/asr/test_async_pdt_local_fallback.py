from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import ghoshell_moss_contrib.asr.async_states as async_states
from ghoshell_moss_contrib.asr.async_states import AsyncPdtListeningState
from ghoshell_moss_contrib.asr.concepts.listener import Recognition


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


@pytest.fixture(autouse=True)
def _disable_latency_log(monkeypatch) -> None:
    monkeypatch.setattr(async_states, "_latency_log", lambda *args, **kwargs: None)


def test_empty_text_commit_policy_keeps_short_fast_and_long_patient(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_EMPTY_TEXT_COMMIT_SECONDS", "0.75")
    monkeypatch.setenv("MOSS_ASR_LONG_EMPTY_TEXT_COMMIT_SECONDS", "1.8")
    monkeypatch.setenv("MOSS_ASR_STABLE_SHORT_MIN_QUIET_SECONDS", "0.35")
    monkeypatch.setenv("MOSS_ASR_LONG_EMPTY_TEXT_MIN_QUIET_SECONDS", "1.2")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert state._empty_text_commit_policy("小白你好") == (0.75, "short", 0.35)
    assert state._empty_text_commit_policy("小白你好。") == (0.75, "punct", 0.25)
    assert state._empty_text_commit_policy("小白我现在要说一个比较长的问题") == (1.8, "long", 1.2)
    assert state._empty_text_commit_policy("小白我现在要说一个比较长的问题。") == (1.8, "long", 1.2)


def test_incomplete_prefix_guard_waits_for_opening_fragment(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_INCOMPLETE_PREFIX_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_INCOMPLETE_PREFIX_MIN_QUIET_SECONDS", "2.4")
    monkeypatch.setenv("MOSS_ASR_INCOMPLETE_PREFIX_MIN_CHARS", "8")
    monkeypatch.setenv("MOSS_VOICE_ADDRESS_WORDS", "小白")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert state._incomplete_prefix_quiet_remaining("小白你好", 0.8) == 0.0
    assert state._incomplete_prefix_quiet_remaining("小白你现在能做什么", 0.8) == 0.0
    assert state._incomplete_prefix_quiet_remaining("小白", 0.8) == pytest.approx(1.6)
    assert state._incomplete_prefix_quiet_remaining("小白请用", 0.8) == pytest.approx(1.6)
    assert state._incomplete_prefix_quiet_remaining("小白。", 2.4) == 0.0
    assert state._incomplete_prefix_quiet_remaining("小白，我想测试一下", 1.17) == pytest.approx(1.23)
    assert state._incomplete_prefix_quiet_remaining("小白，我想测试一下", 2.4) == 0.0
    assert state._incomplete_prefix_quiet_remaining("小白，我想测试一下。", 0.8) == 0.0
    assert state._incomplete_prefix_quiet_remaining("小白，请你听我完整说完这段。", 0.8) == pytest.approx(1.6)
    assert state._incomplete_prefix_quiet_remaining("小白，请你听我完整说完这段。", 2.4) == 0.0
    assert state._incomplete_prefix_quiet_remaining("小白，我现在要说一个长句子。", 0.8) == pytest.approx(1.6)
    assert state._incomplete_prefix_quiet_remaining("小白，我现在要说一个长句子。", 2.4) == 0.0


def test_empty_long_turn_guard_waits_for_long_quiet(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LONG_EMPTY_TEXT_COMMIT_SECONDS", "1.8")
    monkeypatch.setenv("MOSS_ASR_LONG_EMPTY_TEXT_MIN_QUIET_SECONDS", "1.2")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "2.4")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert state._empty_long_turn_quiet_remaining("", 2.0, 0.9) == 0.0
    assert state._empty_long_turn_quiet_remaining("", 3.0, 0.9) == pytest.approx(0.3)
    assert state._empty_long_turn_quiet_remaining("", 3.0, 1.2) == 0.0
    assert state._empty_long_turn_quiet_remaining("小白你好", 3.0, 0.9) == 0.0


def test_speech_no_text_rotate_waits_for_quiet_window(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_SPEECH_NO_TEXT_MAX_SECONDS", "3.2")
    monkeypatch.setenv("MOSS_ASR_SPEECH_NO_TEXT_MIN_QUIET_SECONDS", "0.65")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert not state._speech_no_text_ready_to_rotate(5.2, 0.2)
    assert not state._speech_no_text_ready_to_rotate(3.1, 1.0)
    assert state._speech_no_text_ready_to_rotate(3.2, 0.65)


def test_final_open_fallback_skips_latency_probe_fragments(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MIN_RMS", "500")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert state._should_try_final_open_fallback("请用一句话回答", max_rms=2000)
    assert not state._should_try_final_open_fallback(
        "小白，请用一句话简单回答，你今天最喜欢什么颜色？为什么",
        max_rms=2000,
    )
    assert not state._should_try_final_open_fallback("一定等我这句话", max_rms=2000)
    assert not state._should_try_final_open_fallback("请你一定等我这句话全部说完以后", max_rms=2000)


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
    assert callback.saved_batches == []


@pytest.mark.asyncio
async def test_empty_wake_fallback_recovers_low_rms_empty_short_wake(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "5000")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_QUIET_SECONDS", "0.2")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MAX_SPEECH_SECONDS", "6")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MAX_ACTIVE_SECONDS", "2.4")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MODEL", "tiny")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_PROMPT", "小白你好")
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

    async def _try_local_fallback(**kwargs) -> str:
        fallback_calls.append(kwargs)
        return "小白你好"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 1200, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert batch.commits == 0
    assert fallback_calls
    assert fallback_calls[0]["reason"] == "empty_wake_fallback"
    assert fallback_calls[0]["model_name"] == "tiny"
    assert fallback_calls[0]["initial_prompt"] == "小白你好"
    assert callback.recognitions[-1].text == "小白你好"
    assert callback.recognitions[-1].commit_reason == "empty_wake_fallback"


@pytest.mark.asyncio
async def test_empty_wake_fallback_rejects_unsafe_text(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "5000")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_QUIET_SECONDS", "0.2")
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

    async def _try_local_fallback(**kwargs) -> str:
        fallback_calls.append(kwargs)
        return ""

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 1200, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert fallback_calls
    assert batch.commits == 1
    assert state._commit_reason == "energy_vad"
    assert callback.recognitions == []


def test_empty_wake_fallback_skips_long_active_span(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MAX_ACTIVE_SECONDS", "2.4")

    clock = _Clock()
    clock.now = 1004.0
    monkeypatch.setattr(async_states.time, "time", clock.time)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(clock),
    )
    state._batch_started_at = 1000.0

    eligible, ready, elapsed, last_loud_age, active_span = state._empty_wake_fallback_window(
        has_speech=True,
        attempted=False,
        max_rms=1200,
        first_loud_audio_time=1000.0,
        last_loud_audio_time=1003.0,
    )

    assert not eligible
    assert not ready
    assert elapsed == pytest.approx(4.0)
    assert last_loud_age == pytest.approx(1.0)
    assert active_span == pytest.approx(3.0)


def test_empty_wake_fallback_allows_high_energy_extended_span(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MAX_SPEECH_SECONDS", "6.0")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MAX_ACTIVE_SECONDS", "2.4")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_EXTENDED_MIN_RMS", "5000")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_EXTENDED_MAX_SPEECH_SECONDS", "7.2")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_EXTENDED_MAX_ACTIVE_SECONDS", "6.2")

    clock = _Clock()
    clock.now = 1006.7
    monkeypatch.setattr(async_states.time, "time", clock.time)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(clock),
    )
    state._batch_started_at = 1000.0

    eligible, ready, elapsed, last_loud_age, active_span = state._empty_wake_fallback_window(
        has_speech=True,
        attempted=False,
        max_rms=6000,
        first_loud_audio_time=1000.0,
        last_loud_audio_time=1005.4,
    )

    assert eligible
    assert ready
    assert elapsed == pytest.approx(6.7)
    assert last_loud_age == pytest.approx(1.3)
    assert active_span == pytest.approx(5.4)
    assert state._empty_wake_fallback_audio_seconds(6000) == pytest.approx(4.5)


def test_empty_wake_fallback_extended_audio_window_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_EXTENDED_MIN_RMS", "5000")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MAX_AUDIO_SECONDS", "4.5")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_EXTENDED_MAX_AUDIO_SECONDS", "6.5")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert state._empty_wake_fallback_audio_seconds(4999) == pytest.approx(4.5)
    assert state._empty_wake_fallback_audio_seconds(5000) == pytest.approx(6.5)


def test_empty_wake_fallback_batch_error_requires_extended_energy(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_EXTENDED_MIN_RMS", "5000")

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    assert not state._empty_wake_fallback_allows_batch_error(4999)
    assert state._empty_wake_fallback_allows_batch_error(5000)


def test_empty_wake_fallback_skips_single_loud_spike(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_EMPTY_WAKE_FALLBACK_MIN_ACTIVE_SECONDS", "0.25")

    clock = _Clock()
    clock.now = 1001.0
    monkeypatch.setattr(async_states.time, "time", clock.time)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(clock),
    )
    state._batch_started_at = 1000.0

    eligible, ready, elapsed, last_loud_age, active_span = state._empty_wake_fallback_window(
        has_speech=True,
        attempted=False,
        max_rms=2300,
        first_loud_audio_time=1000.2,
        last_loud_audio_time=1000.2,
    )

    assert not eligible
    assert not ready
    assert elapsed == pytest.approx(0.8)
    assert last_loud_age == pytest.approx(0.8)
    assert active_span == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_unsafe_short_local_fallback_retries_before_drop(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_UNSAFE_DROP_MIN_SECONDS", "2")
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
        return "" if len(fallback_calls) == 1 else "小白你好"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert len(fallback_calls) == 2
    assert callback.recognitions[-1].text == "小白你好"
    assert callback.recognitions[-1].commit_reason == "local_whisper_quiet"
    assert callback.saved_batches == []


@pytest.mark.asyncio
async def test_unsafe_short_cloud_fragment_runs_local_rescue_before_commit(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_RESCUE_UNSAFE_SHORT", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_INPUT_GATE_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_STABLE_TEXT_COMMIT_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_STABLE_SHORT_MIN_QUIET_SECONDS", "0")
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
        vad=None,
    )
    state._current_batch = batch
    state._batch_started_at = clock.time()
    state._last_non_empty_text = "你"
    state._last_non_empty_recognition_time = clock.time() - 1
    state._last_text_change_time = clock.time() - 1

    fallback_calls = []

    async def _try_local_fallback(*, reason: str, max_rms: float, last_loud_age: float) -> str:
        fallback_calls.append((reason, max_rms, last_loud_age))
        return "小白你好"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)

    await state._process_audio_batch(deque([loud]))

    assert fallback_calls
    assert fallback_calls[0][0] == "unsafe_short_text"
    assert batch.commits == 0
    assert callback.recognitions[-1].text == "小白你好"
    assert callback.recognitions[-1].commit_reason == "unsafe_short_text"


@pytest.mark.asyncio
async def test_high_energy_unsafe_local_fallback_commits_for_server_final(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_COMMIT_UNSAFE_MIN_RMS", "1500")
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
        return ""

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert len(fallback_calls) == 1
    assert batch.commits == 1
    assert state._commit_reason == "local_fallback_no_safe_text"
    assert len(callback.saved_batches) == 1
    assert callback.saved_batches[0][0].commit_reason == "local_fallback_no_safe_text"


@pytest.mark.asyncio
async def test_auto_commit_propagates_audio_max_rms_to_final_recognition() -> None:
    callback = _Callback()
    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=callback,
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )
    state._commit_reason = "energy_vad"
    state._commit_audio_max_rms = 2345.6

    await state.on_recognition(Recognition(text="你好", is_last=True))

    assert callback.recognitions[-1].commit_reason == "energy_vad"
    assert callback.recognitions[-1].audio_max_rms == 2345.6


@pytest.mark.asyncio
async def test_safe_local_fallback_audio_save_is_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_SAVE_SAFE_LOCAL_FALLBACK_AUDIO", "1")

    callback = _Callback()
    batch = _Batch()
    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=callback,
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )
    state._current_batch = batch
    state._batch_id = "batch-1"
    state._seq = 41
    batch.buffered.append(np.full(1600, 2000, dtype=np.int16))

    await state._finish_with_local_fallback(
        "小白你好",
        reason="local_whisper_quiet",
        max_rms=2000,
        last_loud_age=0.3,
    )

    assert callback.recognitions[-1].text == "小白你好"
    assert len(callback.saved_batches) == 1
    rec, audio = callback.saved_batches[0]
    assert rec.text == "小白你好"
    assert rec.commit_reason == "local_fallback_safe_text"
    assert len(audio) == 1600


@pytest.mark.asyncio
async def test_rescue_model_accepts_safe_second_pass(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_RESCUE_MODEL", "base")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_CACHE_DIR", "")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_SITE_PACKAGES", "")

    calls = []

    def transcribe(audio, *, sample_rate, model_name, cache_dir, site_packages, initial_prompt):
        calls.append(model_name)
        return "小丸你好" if model_name == "tiny" else "小白你好"

    monkeypatch.setattr(async_states, "_local_whisper_transcribe", transcribe)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )
    batch = _Batch()
    batch.buffered.append(np.full(1600, 2000, dtype=np.int16))
    state._current_batch = batch

    text = await state._try_local_fallback(
        reason="local_whisper_quiet",
        max_rms=2000,
        last_loud_age=0.3,
    )

    assert text == "小白你好"
    assert calls == ["tiny", "base"]


@pytest.mark.asyncio
async def test_rescue_model_rejects_unsafe_second_pass(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_RESCUE_MODEL", "base")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_CACHE_DIR", "")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_SITE_PACKAGES", "")

    calls = []

    def transcribe(audio, *, sample_rate, model_name, cache_dir, site_packages, initial_prompt):
        calls.append(model_name)
        return "小丸你好" if model_name == "tiny" else "小完一号"

    monkeypatch.setattr(async_states, "_local_whisper_transcribe", transcribe)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )
    batch = _Batch()
    batch.buffered.append(np.full(1600, 2000, dtype=np.int16))
    state._current_batch = batch

    text = await state._try_local_fallback(
        reason="local_whisper_quiet",
        max_rms=2000,
        last_loud_age=0.3,
    )

    assert text == ""
    assert calls == ["tiny", "base"]


@pytest.mark.asyncio
async def test_local_fallback_rejects_reversed_wake_fragment(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_RESCUE_PROMPT", "小白你好")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_CACHE_DIR", "")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_SITE_PACKAGES", "")

    calls = []

    def transcribe(audio, *, sample_rate, model_name, cache_dir, site_packages, initial_prompt):
        calls.append(initial_prompt or "")
        return "好,小白" if len(calls) == 1 else "小白你好"

    monkeypatch.setattr(async_states, "_local_whisper_transcribe", transcribe)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )
    batch = _Batch()
    batch.buffered.append(np.full(1600, 2000, dtype=np.int16))
    state._current_batch = batch

    text = await state._try_local_fallback(
        reason="local_whisper_quiet",
        max_rms=2000,
        last_loud_age=0.9,
    )

    assert text == ""
    assert calls == [""]


@pytest.mark.asyncio
async def test_open_local_fallback_accepts_addressed_question(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_CACHE_DIR", "")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_SITE_PACKAGES", "")

    calls = []

    def transcribe(audio, *, sample_rate, model_name, cache_dir, site_packages, initial_prompt):
        calls.append(model_name)
        return "小白晴,用一句话简单回答你,今天最喜欢什么颜色,为什么?"

    monkeypatch.setattr(async_states, "_local_whisper_transcribe", transcribe)

    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=_Callback(),
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )
    batch = _Batch()
    batch.buffered.append(np.full(1600, 3000, dtype=np.int16))
    state._current_batch = batch

    text = await state._try_local_fallback(
        reason="final_open_fallback",
        max_rms=3000,
        last_loud_age=1.0,
        allow_open_request=True,
        model_name="base",
        timeout_seconds=3.5,
        max_audio_seconds=8.0,
    )

    assert text.startswith("小白请用一句话")
    assert "颜色" in text
    assert calls == ["base"]


@pytest.mark.asyncio
async def test_final_open_fallback_recovers_empty_high_energy(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0.2")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "0.1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MODEL", "base")
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
    state._batch_started_at = clock.time() - 1.0

    fallback_calls = []

    async def _try_local_fallback(**kwargs) -> str:
        fallback_calls.append(kwargs)
        return "小白请简单回答机器人为什么需要耳朵"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert fallback_calls
    assert fallback_calls[0]["reason"] == "final_open_fallback"
    assert fallback_calls[0]["model_name"] == "base"
    assert batch.commits == 0
    assert callback.recognitions[-1].text == "小白请简单回答机器人为什么需要耳朵"
    assert callback.recognitions[-1].commit_reason == "final_open_fallback"


@pytest.mark.asyncio
async def test_final_open_fallback_runs_before_no_safe_text_commit(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_COMMIT_UNSAFE_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MODEL", "base")
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

    async def _try_local_fallback(**kwargs) -> str:
        fallback_calls.append(kwargs)
        if kwargs["reason"] == "local_whisper_quiet":
            return ""
        return "小白请简短回答你最喜欢做什么"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert [call["reason"] for call in fallback_calls] == [
        "local_whisper_quiet",
        "final_open_fallback",
    ]
    assert batch.commits == 0
    assert callback.recognitions[-1].text == "小白请简短回答你最喜欢做什么"
    assert callback.recognitions[-1].commit_reason == "final_open_fallback"


@pytest.mark.asyncio
async def test_final_open_fallback_accepts_safe_short_wake(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", "10")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_COMMIT_UNSAFE_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MODEL", "base")
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

    async def _try_local_fallback(**kwargs) -> str:
        if kwargs["reason"] == "local_whisper_quiet":
            return ""
        return "小白你好"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)
    quiet = np.zeros(1600, dtype=np.int16)

    await state._process_audio_batch(deque([loud, quiet]))

    assert batch.commits == 0
    assert callback.recognitions[-1].text == "小白你好"
    assert callback.recognitions[-1].commit_reason == "final_open_fallback"


@pytest.mark.asyncio
async def test_final_open_fallback_recovers_mid_fragment(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_LOCAL_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MIN_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_FINAL_OPEN_FALLBACK_MODEL", "base")
    monkeypatch.setenv("MOSS_ASR_INPUT_GATE_RMS", "500")
    monkeypatch.setenv("MOSS_ASR_STABLE_TEXT_COMMIT_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_STABLE_TEXT_MIN_QUIET_SECONDS", "0")
    monkeypatch.setenv("MOSS_ASR_STABLE_SHORT_MIN_QUIET_SECONDS", "0")
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
        vad=None,
    )
    state._current_batch = batch
    state._batch_started_at = clock.time()
    state._last_non_empty_text = "请用一句话"
    state._last_non_empty_recognition_time = clock.time() - 1
    state._last_text_change_time = clock.time() - 1

    fallback_calls = []

    async def _try_local_fallback(**kwargs) -> str:
        fallback_calls.append(kwargs)
        return "小白请用一句话简单回答你今天最喜欢什么颜色为什么"

    state._try_local_fallback = _try_local_fallback

    loud = np.full(1600, 2000, dtype=np.int16)

    await state._process_audio_batch(deque([loud]))

    assert fallback_calls
    assert fallback_calls[0]["reason"] == "final_open_fallback"
    assert batch.commits == 0
    assert callback.recognitions[-1].text.startswith("小白请用一句话")
    assert callback.recognitions[-1].commit_reason == "final_open_fallback"


@pytest.mark.asyncio
async def test_late_cloud_final_after_local_fallback_is_dropped() -> None:
    callback = _Callback()
    state = AsyncPdtListeningState(
        recognizer=SimpleNamespace(sample_rate=16000, frame_duration=0.1),
        audio_input=SimpleNamespace(),
        callback=callback,
        logger=_Logger(),
        vad=_CommitOnQuietVad(_Clock()),
    )

    await state._finish_with_local_fallback(
        "小白请简单回答机器人为什么需要耳朵",
        reason="final_open_fallback",
        max_rms=3000,
        last_loud_age=1.0,
    )
    await state.on_recognition(
        Recognition(
            text="小白，请简单回答机器人为什么需要耳朵",
            is_last=True,
        )
    )

    assert len(callback.recognitions) == 1
    assert callback.recognitions[0].text == "小白请简单回答机器人为什么需要耳朵"
