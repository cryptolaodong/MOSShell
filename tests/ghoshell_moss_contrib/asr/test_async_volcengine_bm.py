import numpy as np
import pytest

from ghoshell_moss_contrib.asr.async_volcengine_bm import AsyncVocEngineBigModelStreamASRBatch
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
