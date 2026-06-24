import numpy as np
import pytest

from ghoshell_moss_contrib.asr.async_volcengine_bm import (
    AsyncLocalOnlyRecognitionBatch,
    AsyncVocEngineBigModelASR,
    AsyncVocEngineBigModelStreamASRBatch,
    _mark_asr_cloud_failure,
    _reset_asr_cloud_circuit_for_tests,
)
from ghoshell_moss_contrib.asr.volcengine_bm_protocol import VolcanoBigModelASRConfig


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
    async def on_recognition(self, result) -> None:
        pass

    async def on_error(self, error: str) -> None:
        raise AssertionError(error)

    async def save_batch(self, rec, audio) -> None:
        pass


@pytest.mark.asyncio
async def test_batch_keeps_audio_buffer_before_websocket_connects() -> None:
    batch = AsyncVocEngineBigModelStreamASRBatch(
        batch_id="test",
        config=VolcanoBigModelASRConfig(),
        callback=_Callback(),
        logger=_Logger(),
    )
    audio = np.arange(1600, dtype=np.int16)

    await batch.buffer(audio)

    buffered = await batch.get_buffer()
    assert np.array_equal(buffered, audio)


@pytest.mark.asyncio
async def test_cloud_circuit_uses_local_only_batch_with_audio_buffer(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_CLOUD_CIRCUIT_BREAKER_ENABLED", "1")
    monkeypatch.setenv("MOSS_ASR_CLOUD_CIRCUIT_FAILURE_THRESHOLD", "1")
    monkeypatch.setenv("MOSS_ASR_CLOUD_CIRCUIT_BACKOFF_SECONDS", "30")
    _reset_asr_cloud_circuit_for_tests()
    try:
        _mark_asr_cloud_failure(TimeoutError("timed out during opening handshake"), _Logger())
        recognizer = AsyncVocEngineBigModelASR(
            config=VolcanoBigModelASRConfig(),
            logger=_Logger(),
            callback=_Callback(),
        )

        batch = await recognizer.new_batch(callback=_Callback(), batch_id="local-only")
        assert isinstance(batch, AsyncLocalOnlyRecognitionBatch)

        audio = np.arange(800, dtype=np.int16)
        await batch.start()
        await batch.buffer(audio)
        assert not await batch.is_done()

        buffered = await batch.get_buffer()
        assert np.array_equal(buffered, audio)

        await batch.commit()
        await batch.wait_until_done(timeout=0.01)
        assert await batch.is_done()
    finally:
        _reset_asr_cloud_circuit_for_tests()


@pytest.mark.asyncio
async def test_force_local_only_batch_with_audio_buffer(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_FORCE_LOCAL_ONLY", "1")
    _reset_asr_cloud_circuit_for_tests()
    recognizer = AsyncVocEngineBigModelASR(
        config=VolcanoBigModelASRConfig(),
        logger=_Logger(),
        callback=_Callback(),
    )

    batch = await recognizer.new_batch(callback=_Callback(), batch_id="forced-local")
    assert isinstance(batch, AsyncLocalOnlyRecognitionBatch)

    audio = np.arange(800, dtype=np.int16)
    await batch.start()
    await batch.buffer(audio)
    buffered = await batch.get_buffer()
    assert np.array_equal(buffered, audio)
