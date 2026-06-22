"""音频输入实现 — ReachyMini WebRTC 麦克风 + SoundDevice 本地麦克风。"""
import asyncio
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

__all__ = ['ReachyMicAudioInput', 'SoundDeviceAudioInput']


class ReachyMicAudioInput:
    """通过 ReachyMini WebRTC 获取机器人麦克风音频（异步接口）。"""

    def __init__(
        self,
        rate: int = 16000,
        channels: int = 1,
        dtype: np.dtype = np.dtype(np.int16),
        logger=None,
    ):
        self.input_id = "reachy_mic"
        self.rate = rate
        self.channels = channels
        self.dtype = dtype
        self._logger = logger or logging.getLogger("ReachyMicAudioInput")
        self._mini = None
        self._closed = False
        self._started = False
        self._read_count = 0
        self._warmup_reads = 20  # 前20次read返回静音，等WebRTC稳定

    async def start(self) -> None:
        if self._closed or self._started:
            return
        os.environ['GST_PLUGIN_PATH'] = ''
        os.environ['GST_PLUGIN_SYSTEM_PATH'] = ''
        from reachy_mini import ReachyMini
        robot_host = os.environ.get('REACHY_ROBOT_HOST', 'reachy-mini.local')
        self._logger.warning("ReachyMicAudioInput: connecting to %s ...", robot_host)
        print(f"[ReachyMicAudioInput] connecting to {robot_host}...", flush=True)
        try:
            _orig = ReachyMini.release_media
            ReachyMini.release_media = lambda self: None
            self._mini = ReachyMini(host=robot_host, connection_mode="network")
            ReachyMini.release_media = _orig
            self._started = True
            print(f"[ReachyMicAudioInput] SDK connected, waiting WebRTC...", flush=True)
            await asyncio.sleep(5)
            sr = self._mini.media.get_input_audio_samplerate()
            print(f"[ReachyMicAudioInput] ready! sr={sr}", flush=True)
            self._logger.warning("ReachyMicAudioInput: ready! sr=%s", sr)
        except Exception as e:
            print(f"[ReachyMicAudioInput] FAILED: {e}", flush=True)
            self._logger.error("ReachyMicAudioInput: connect failed: %s", e)
            raise

    async def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        if not self._started or self._mini is None:
            return np.zeros(int(self.rate * (duration or 0.1)), dtype=self.dtype)
        try:
            target_samples = int(self.rate * (duration or 0.1))
            # 预热期：前N次读取返回静音，避免 WebRTC 初始噪声误触发 VAD
            if self._warmup_reads > 0:
                self._warmup_reads -= 1
                # 消费掉音频数据但不返回（让 WebRTC buffer 排空）
                for _ in range(target_samples // 320 + 1):
                    self._mini.media.get_audio_sample()
                    await asyncio.sleep(0.005)
                return np.zeros(target_samples, dtype=self.dtype)
            chunks = []
            collected = 0
            max_attempts = target_samples // 160 + 10
            for _ in range(max_attempts):
                sample = self._mini.media.get_audio_sample()
                if sample is None:
                    await asyncio.sleep(0.01)
                    continue
                if sample.ndim > 1:
                    mono = sample[:, 0]
                else:
                    mono = sample
                if mono.dtype == np.float32:
                    mono = (mono * 32767).astype(np.int16)
                chunks.append(mono)
                collected += len(mono)
                if collected >= target_samples:
                    break
                await asyncio.sleep(0.005)
            if not chunks:
                return np.zeros(target_samples, dtype=self.dtype)
            result = np.concatenate(chunks)
            if len(result) > target_samples:
                result = result[:target_samples]
            self._read_count += 1
            if self._read_count <= 3 or self._read_count % 100 == 0:
                rms = float(np.sqrt(np.mean(result.astype(float) ** 2)))
                print(f"[ReachyMicAudioInput] read #{self._read_count}: {len(result)} samples, rms={rms:.1f}", flush=True)
            return result
        except Exception as e:
            self._logger.debug("ReachyMicAudioInput read error: %s", e)
            return np.zeros(int(self.rate * (duration or 0.1)), dtype=self.dtype)

    async def stop(self) -> None:
        pass

    async def close(self, error: Optional[Exception] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._mini:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:
                pass
            self._mini = None

    def closed(self) -> bool:
        return self._closed


class SoundDeviceAudioInput:
    """使用 sounddevice 库采集本地麦克风音频（异步接口）。"""

    def __init__(self, rate: int = 16000, channels: int = 1, dtype=None, logger=None):
        self.input_id = "sounddevice"
        self.rate = rate
        self.channels = channels
        self.dtype = dtype or np.dtype(np.int16)
        self._logger = logger or logging.getLogger("SoundDeviceAudioInput")
        self._stream = None
        self._closed = False
        self._buffer = []
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._buffer.append(indata.copy())

    async def start(self) -> None:
        if self._closed or self._stream is not None:
            return
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.rate, channels=self.channels,
            dtype=self.dtype, callback=self._callback,
            blocksize=int(self.rate * 0.1),
        )
        self._stream.start()

    async def read(self, *, rate=None, duration=None) -> np.ndarray:
        duration = duration or 0.1
        samples_needed = int(self.rate * duration)
        deadline = time.time() + duration + 0.05
        while time.time() < deadline:
            with self._lock:
                total = sum(b.shape[0] for b in self._buffer)
                if total >= samples_needed:
                    break
            await asyncio.sleep(0.01)
        with self._lock:
            if not self._buffer:
                return np.zeros(samples_needed, dtype=self.dtype)
            all_data = np.concatenate(self._buffer)
            self._buffer.clear()
        if len(all_data) > samples_needed:
            result = all_data[:samples_needed]
            with self._lock:
                self._buffer.append(all_data[samples_needed:])
        else:
            result = all_data
        if result.ndim > 1:
            result = result[:, 0]
        return result.flatten()

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def close(self, error=None) -> None:
        if self._closed:
            return
        self._closed = True
        await self.stop()

    def closed(self) -> bool:
        return self._closed
