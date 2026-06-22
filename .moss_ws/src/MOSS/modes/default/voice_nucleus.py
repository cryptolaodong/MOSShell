"""语音感知核 — 持续流式监听机器人麦克风，实时ASR识别。

使用框架的 AsyncVocEngineBigModelASR（流式WebSocket ASR）。
音频持续发送，识别结果实时返回，延迟极低。
"""
import asyncio
import logging
import os
import time
import threading
from typing import Callable, Optional

import numpy as np
from typing_extensions import Self

from ghoshell_moss.core.blueprint.mindflow import Nucleus, Signal, Impulse, Priority
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import Message, Text
from ghoshell_container import IoCContainer


class VoiceNucleus(Nucleus):
    """持续流式监听机器人麦克风的感知核。"""

    def __init__(self, container: IoCContainer, logger: LoggerItf | None = None):
        self._container = container
        self._logger = logger or get_moss_logger()
        self._name = "voice_nucleus"
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._broadcast_cb: Optional[Callable[[Signal], None]] = None
        self._notify_cb: Optional[Callable[[Impulse], None]] = None
        self._suppress_until: float = 0.0
        self._mini = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "持续流式监听机器人麦克风，实时语音识别"

    def status(self) -> str:
        return "listening" if self._running else ""

    def signals(self) -> list:
        return []

    def clear(self) -> None:
        pass

    def add_signal(self, signal: Signal) -> None:
        pass

    def with_bus(self, signal_broadcast, impulse_notify) -> None:
        self._broadcast_cb = signal_broadcast
        self._notify_cb = impulse_notify

    def suppress(self, suppress_by: Impulse) -> None:
        self._suppress_until = time.time() + 2.0

    def pop_impulse(self, impulse: Impulse) -> None:
        pass

    def peek(self, no_stale: bool = True) -> Impulse | None:
        return None

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> Self:
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())
        self._logger.info("[VoiceNucleus] Started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("[VoiceNucleus] Stopped")

    async def _listen_loop(self):
        """流式监听：持续从麦克风获取音频 → 流式发给ASR → 实时收结果。"""
        await asyncio.sleep(15)  # Wait for audio player WebRTC init

        # Connect robot mic
        try:
            from reachy_mini import ReachyMini
            robot_host = os.environ.get('REACHY_ROBOT_HOST', 'reachy-mini.local')
            _orig = ReachyMini.release_media
            ReachyMini.release_media = lambda self: None
            self._mini = ReachyMini(host=robot_host)
            ReachyMini.release_media = _orig
            input_sr = self._mini.media.get_input_audio_samplerate()
            self._logger.info("[VoiceNucleus] Robot mic connected! sr=%s", input_sr)
        except Exception as e:
            self._logger.error("[VoiceNucleus] Failed to connect robot mic: %s", e)
            return

        # Create streaming ASR recognizer (framework's implementation)
        try:
            from ghoshell_moss_contrib.asr.async_volcengine_bm import AsyncVocEngineBigModelASR
            from ghoshell_moss_contrib.asr.volcengine_bm_protocol import VolcanoBigModelASRConfig
            from ghoshell_moss_contrib.asr.async_concepts import AsyncRecognitionCallback, Recognition

            config = VolcanoBigModelASRConfig()
            config.resolve_env()
            self._recognizer = AsyncVocEngineBigModelASR(config=config, logger=self._logger)
            await self._recognizer.start()
            self._logger.info("[VoiceNucleus] Streaming ASR ready")
        except Exception as e:
            self._logger.error("[VoiceNucleus] ASR init failed: %s", e)
            return

        # Recognition callback
        nucleus_self = self

        class VoiceCallback(AsyncRecognitionCallback):
            async def on_recognition(self, result: Recognition):
                if not result.text or not result.text.strip():
                    return
                if result.is_last:
                    text = result.text.strip()
                    nucleus_self._logger.info("[VoiceNucleus] ASR: %s", text)
                    from ghoshell_moss.core.blueprint.matrix import Matrix
                    matrix = nucleus_self._container.get(Matrix)
                    if matrix:
                        matrix.session.add_input_signal(text, description=f"voice: {text[:30]}")
                        nucleus_self._suppress_until = time.time() + 12.0

            async def on_state_change(self, state: str):
                pass

            async def on_error(self, error: str):
                nucleus_self._logger.warning("[VoiceNucleus] ASR error: %s", error)

            async def on_waken(self):
                pass

            async def save_batch(self, rec, audio):
                pass

        callback = VoiceCallback()

        # Main streaming loop: continuously feed audio to ASR
        batch = None
        silence_count = 0
        is_speaking = False

        while self._running:
            try:
                # Get audio from robot mic
                sample = await asyncio.to_thread(self._mini.media.get_audio_sample)
                if sample is None:
                    await asyncio.sleep(0.05)
                    continue

                # Convert to int16
                if sample.dtype == np.float32:
                    sample_i16 = (sample * 32767).astype(np.int16)
                else:
                    sample_i16 = sample.astype(np.int16)
                if sample_i16.ndim > 1:
                    sample_i16 = sample_i16[:, 0]

                # Skip during suppress (echo cancellation)
                if time.time() < self._suppress_until:
                    if batch is not None:
                        await batch.commit()
                        batch = None
                    is_speaking = False
                    silence_count = 0
                    continue

                # Simple VAD
                rms = np.sqrt(np.mean(sample_i16.astype(float) ** 2))

                if rms > 800:  # Speech detected
                    if batch is None:
                        # Start new streaming ASR batch
                        batch = await self._recognizer.new_batch(
                            callback=callback,
                            vad=800,  # server-side VAD in ms
                            stop_on_sentence=True,
                        )
                    await batch.buffer(sample_i16)
                    is_speaking = True
                    silence_count = 0
                elif is_speaking:
                    silence_count += 1
                    if batch is not None:
                        await batch.buffer(sample_i16)
                    if silence_count > 7:  # ~0.7s silence
                        if batch is not None:
                            await batch.commit()
                            # Wait for final result
                            try:
                                await asyncio.wait_for(batch.wait_until_done(timeout=5), timeout=6)
                            except (asyncio.TimeoutError, Exception):
                                pass
                            batch = None
                        is_speaking = False
                        silence_count = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.debug("[VoiceNucleus] Loop: %s", e)
                await asyncio.sleep(0.5)

        # Cleanup
        if batch is not None:
            try:
                await batch.commit()
            except:
                pass
        try:
            await self._recognizer.close()
        except:
            pass
