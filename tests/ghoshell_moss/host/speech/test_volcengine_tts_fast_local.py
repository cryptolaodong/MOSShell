import asyncio
import logging

import numpy as np
import pytest

from ghoshell_moss.contracts.speech import TTSItem
from ghoshell_moss.host.speech.volcengine_tts.tts import (
    VolcengineTTS,
    VolcengineTTSConf,
)


async def _collect_items(batch) -> list[TTSItem]:
    items = []
    async for item in batch.items():
        items.append(item)
    return items


@pytest.mark.asyncio
async def test_fast_local_phrase_batch_emits_cached_audio(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/say" if name == "say" else None)
    tts = VolcengineTTS(
        conf=VolcengineTTSConf(
            fast_local_phrase_tts=True,
            fast_local_phrase_wait=0.01,
        ),
        logger=logging.getLogger("test_fast_local_phrase_batch_emits_cached_audio"),
    )
    tts._running_loop = asyncio.get_running_loop()
    batch = tts._create_batch(batch_id="fast")
    audio = np.arange(16, dtype=np.int16)
    monkeypatch.setattr(tts, "_fast_local_phrase_audio", lambda *_args: audio)

    await batch.start()
    batch.feed("在呢。")
    batch.commit()

    assert await tts._try_fast_local_phrase_batch(batch)
    assert batch.is_closed()
    items = await _collect_items(batch)
    assert len(items) == 1
    assert np.array_equal(items[0]["audio"], audio)


@pytest.mark.asyncio
async def test_fast_local_phrase_batch_ignores_open_text(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/say" if name == "say" else None)
    tts = VolcengineTTS(
        conf=VolcengineTTSConf(
            fast_local_phrase_tts=True,
            fast_local_phrase_wait=0.01,
        ),
        logger=logging.getLogger("test_fast_local_phrase_batch_ignores_open_text"),
    )
    tts._running_loop = asyncio.get_running_loop()
    batch = tts._create_batch(batch_id="slow")

    await batch.start()
    batch.feed("我可以帮你控制动作，也可以聊天。")
    batch.commit()

    assert not await tts._try_fast_local_phrase_batch(batch)
    assert not batch.is_closed()


def test_fast_local_phrase_defaults_include_reachy_deterministic_replies() -> None:
    tts = VolcengineTTS(
        conf=VolcengineTTSConf(),
        logger=logging.getLogger("test_fast_local_phrase_defaults_include_reachy_deterministic_replies"),
    )

    assert "我能听你说话、回答问题，并同步表情和头部动作" in tts._fast_local_phrases
    assert "我会等你说完，再简短回答" in tts._fast_local_phrases
