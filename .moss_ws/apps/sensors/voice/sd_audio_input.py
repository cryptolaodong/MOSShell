"""音频输入实现 — ReachyMini WebRTC 麦克风 + SoundDevice 本地麦克风。"""
import asyncio
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

__all__ = ['ReachyMicAudioInput', 'SoundDeviceAudioInput']


def _is_truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on", "sounddevice", "local"}


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
        self._fallback = None
        self._closed = False
        self._started = False
        self._unavailable = False
        self._warned_unavailable_read = False
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
            fallback = os.environ.get("MOSS_REACHY_MIC_FALLBACK", "sounddevice")
            if _is_truthy(fallback):
                try:
                    self._fallback = SoundDeviceAudioInput(
                        rate=self.rate,
                        channels=self.channels,
                        dtype=self.dtype,
                        logger=self._logger,
                        raise_on_unavailable=True,
                    )
                    await self._fallback.start()
                    self._started = True
                    print("[ReachyMicAudioInput] using local sounddevice fallback", flush=True)
                    self._logger.warning("ReachyMicAudioInput: using local sounddevice fallback")
                    return
                except Exception as fallback_error:
                    print(f"[ReachyMicAudioInput] local fallback unavailable: {fallback_error}", flush=True)
                    self._logger.error("ReachyMicAudioInput: fallback mic failed: %s", fallback_error)
            self._started = True
            self._unavailable = True
            print(
                "[ReachyMicAudioInput] no microphone source available; voice input will read silence",
                flush=True,
            )
            self._logger.warning("ReachyMicAudioInput: unavailable, returning silence")

    async def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        if self._fallback is not None:
            return await self._fallback.read(rate=rate, duration=duration)
        if self._unavailable:
            if not self._warned_unavailable_read:
                print("[ReachyMicAudioInput] read silence: microphone source is unavailable", flush=True)
                self._warned_unavailable_read = True
            return np.zeros(int(self.rate * (duration or 0.1)), dtype=self.dtype)
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
        if self._fallback is not None:
            await self._fallback.stop()
        pass

    async def close(self, error: Optional[Exception] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fallback is not None:
            await self._fallback.close(error)
            self._fallback = None
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

    def __init__(
        self,
        rate: int = 16000,
        channels: int = 1,
        dtype=None,
        logger=None,
        raise_on_unavailable: bool = False,
    ):
        self.input_id = "sounddevice"
        self.rate = rate
        self.channels = channels
        self.dtype = dtype or np.dtype(np.int16)
        self._logger = logger or logging.getLogger("SoundDeviceAudioInput")
        self._raise_on_unavailable = raise_on_unavailable
        self._stream = None
        self._closed = False
        self._unavailable = False
        self._warned_unavailable_read = False
        self._buffer = []
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._buffer.append(indata.copy())

    def _format_devices(self, sd) -> str:
        try:
            devices = sd.query_devices()
        except Exception as e:
            return f"failed to query devices: {e}"
        rows = []
        for index, device in enumerate(devices):
            rows.append(
                f"{index}:{device.get('name')} "
                f"in={device.get('max_input_channels')} out={device.get('max_output_channels')}"
            )
        return "; ".join(rows) or "no CoreAudio devices"

    def _select_device(self, sd) -> Optional[int]:
        requested = os.environ.get("MOSS_SOUNDDEVICE_INPUT_DEVICE", "").strip()
        devices = sd.query_devices()
        input_devices = [
            (index, device)
            for index, device in enumerate(devices)
            if int(device.get("max_input_channels", 0) or 0) > 0
        ]
        if requested:
            if requested.isdigit():
                index = int(requested)
                device = sd.query_devices(index)
                if int(device.get("max_input_channels", 0) or 0) <= 0:
                    raise RuntimeError(
                        f"MOSS_SOUNDDEVICE_INPUT_DEVICE={index} has no input channels. "
                        f"Devices: {self._format_devices(sd)}"
                    )
                return index
            needle = requested.lower()
            for index, device in input_devices:
                if needle in str(device.get("name", "")).lower():
                    return index
            raise RuntimeError(
                f"MOSS_SOUNDDEVICE_INPUT_DEVICE={requested!r} did not match an input device. "
                f"Devices: {self._format_devices(sd)}"
            )
        if not input_devices:
            raise RuntimeError(f"no input-capable microphone found. Devices: {self._format_devices(sd)}")
        default_input = sd.default.device[0]
        if isinstance(default_input, int) and default_input >= 0:
            try:
                default_device = sd.query_devices(default_input)
                if int(default_device.get("max_input_channels", 0) or 0) > 0:
                    return default_input
            except Exception:
                pass
        return input_devices[0][0]

    async def start(self) -> None:
        if self._closed or self._stream is not None:
            return
        import sounddevice as sd
        try:
            device = self._select_device(sd)
            device_info = sd.query_devices(device)
            print(
                f"[SoundDeviceAudioInput] using device {device}: {device_info.get('name')} "
                f"({device_info.get('max_input_channels')} in)",
                flush=True,
            )
            self._stream = sd.InputStream(
                samplerate=self.rate, channels=self.channels,
                dtype=self.dtype, callback=self._callback,
                device=device,
                blocksize=int(self.rate * 0.1),
            )
            self._stream.start()
        except Exception as e:
            self._unavailable = True
            print(f"[SoundDeviceAudioInput] unavailable: {e}", flush=True)
            self._logger.error("SoundDeviceAudioInput unavailable: %s", e)
            if self._raise_on_unavailable:
                raise

    async def read(self, *, rate=None, duration=None) -> np.ndarray:
        duration = duration or 0.1
        samples_needed = int(self.rate * duration)
        if self._unavailable:
            if not self._warned_unavailable_read:
                print("[SoundDeviceAudioInput] read silence: local microphone is unavailable", flush=True)
                self._warned_unavailable_read = True
            return np.zeros(samples_needed, dtype=self.dtype)
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
