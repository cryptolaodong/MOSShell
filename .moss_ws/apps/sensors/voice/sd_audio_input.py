"""音频输入实现 — ReachyMini WebRTC 麦克风 + SoundDevice 本地麦克风。"""
import asyncio
import json
import logging
import os
import threading
import time
from typing import Optional
from urllib import request

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
        self._media = None
        self._MediaBackend = None
        self._MediaManager = None
        self._get_producer_list = None
        self._daemon_url = ""
        self._signalling_host = ""
        self._camera_specs = None
        self._fallback = None
        self._closed = False
        self._started = False
        self._unavailable = False
        self._warned_unavailable_read = False
        self._read_count = 0
        self._warmup_reads = 20  # 前20次read返回静音，等WebRTC稳定
        self._last_media_check = 0.0
        self._media_check_interval = float(os.environ.get("MOSS_REACHY_MIC_MEDIA_CHECK_SECONDS", "2.0"))
        self._last_nonzero_read = time.monotonic()
        self._zero_read_count = 0
        self._zero_reopen_reads = int(os.environ.get("MOSS_REACHY_MIC_ZERO_REOPEN_READS", "20"))
        self._zero_reopen_seconds = float(os.environ.get("MOSS_REACHY_MIC_ZERO_REOPEN_SECONDS", "2.0"))
        self._zero_fallback_reads = int(os.environ.get("MOSS_REACHY_MIC_ZERO_FALLBACK_READS", "30"))
        self._recovering_media = False

    async def start(self) -> None:
        if self._closed or self._started:
            return
        os.environ['GST_PLUGIN_PATH'] = ''
        os.environ['GST_PLUGIN_SYSTEM_PATH'] = ''

        from reachy_mini.media.camera_constants import get_camera_specs_by_name
        from reachy_mini.media.media_manager import MediaBackend, MediaManager
        from reachy_mini.media.webrtc_utils import get_producer_list
        robot_host = os.environ.get('REACHY_ROBOT_HOST', 'reachy-mini.local')
        daemon_url = f"http://{robot_host}:8000"
        self._MediaBackend = MediaBackend
        self._MediaManager = MediaManager
        self._get_producer_list = get_producer_list
        self._daemon_url = daemon_url
        self._logger.warning("ReachyMicAudioInput: connecting to %s ...", robot_host)
        print(f"[ReachyMicAudioInput] connecting to {robot_host}...", flush=True)
        try:
            status = {}
            try:
                status = self._fetch_daemon_status(timeout=3)
                print(
                    "[ReachyMicAudioInput] daemon status "
                    f"state={status.get('state')} media_released={status.get('media_released')} "
                    f"no_media={status.get('no_media')} error={status.get('error')}",
                    flush=True,
                )
                self._logger.warning(
                    "ReachyMicAudioInput: daemon status state=%s media_released=%s no_media=%s error=%s",
                    status.get("state"),
                    status.get("media_released"),
                    status.get("no_media"),
                    status.get("error"),
                )
                if status.get("media_released"):
                    print("[ReachyMicAudioInput] acquiring daemon media before WebRTC...", flush=True)
                    self._acquire_daemon_media(timeout=10)
            except Exception as acquire_error:
                print(f"[ReachyMicAudioInput] media acquire preflight failed: {acquire_error}", flush=True)
                self._logger.warning("ReachyMicAudioInput: media acquire preflight failed: %s", acquire_error)

            signalling_host = str(status.get("wlan_ip") or robot_host)
            self._signalling_host = signalling_host
            if not await self._wait_for_producer(signalling_host):
                raise RuntimeError(f"Reachy WebRTC producer not ready at {signalling_host}:8443")

            specs_name = str(status.get("camera_specs_name") or "")
            self._camera_specs = get_camera_specs_by_name(specs_name) if specs_name else None
            self._open_media(reason="startup", warmup_reads=self._warmup_reads)
            self._started = True
            print(f"[ReachyMicAudioInput] WebRTC media connected, warming up...", flush=True)
            await asyncio.sleep(float(os.environ.get("MOSS_REACHY_MIC_WEBRTC_WARMUP_SECONDS", "1.5")))
            sr = self._media.get_input_audio_samplerate()
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

    def _fallback_enabled(self) -> bool:
        return _is_truthy(os.environ.get("MOSS_REACHY_MIC_FALLBACK", "sounddevice"))

    async def _switch_to_sounddevice_fallback(self, *, reason: str) -> bool:
        if self._fallback is not None:
            return True
        if not self._fallback_enabled():
            return False
        try:
            fallback = SoundDeviceAudioInput(
                rate=self.rate,
                channels=self.channels,
                dtype=self.dtype,
                logger=self._logger,
                raise_on_unavailable=True,
            )
            await fallback.start()
            self._fallback = fallback
            self._zero_read_count = 0
            print(f"[ReachyMicAudioInput] switched to local sounddevice fallback reason={reason}", flush=True)
            self._logger.warning("ReachyMicAudioInput: switched to local sounddevice fallback reason=%s", reason)
            return True
        except Exception as fallback_error:
            print(f"[ReachyMicAudioInput] local fallback unavailable after {reason}: {fallback_error}", flush=True)
            self._logger.error("ReachyMicAudioInput: local fallback unavailable after %s: %s", reason, fallback_error)
            return False

    def _fetch_daemon_status(self, *, timeout: float = 0.7) -> dict:
        with request.urlopen(f"{self._daemon_url}/api/daemon/status", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _acquire_daemon_media(self, *, timeout: float = 2.0) -> None:
        req = request.Request(f"{self._daemon_url}/api/media/acquire", method="POST")
        with request.urlopen(req, timeout=timeout):
            pass

    async def _wait_for_producer(self, signalling_host: str, *, timeout: Optional[float] = None) -> bool:
        deadline = time.monotonic() + float(
            timeout if timeout is not None else os.environ.get("MOSS_REACHY_MIC_WEBRTC_READY_TIMEOUT", "8")
        )
        while time.monotonic() < deadline:
            try:
                producers = self._get_producer_list(signalling_host, 8443)
                if any(meta.get("name") == "reachymini" for meta in producers.values()):
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.25)
        return False

    def _open_media(self, *, reason: str, warmup_reads: Optional[int] = None) -> None:
        if self._media is not None:
            try:
                self._media.close()
            except Exception:
                pass
        self._media = self._MediaManager(
            backend=self._MediaBackend.WEBRTC,
            log_level="WARNING",
            signalling_host=self._signalling_host,
            camera_specs=self._camera_specs,
            daemon_url=self._daemon_url,
        )
        self._warmup_reads = int(warmup_reads if warmup_reads is not None else os.environ.get("MOSS_REACHY_MIC_REOPEN_WARMUP_READS", "5"))
        self._last_nonzero_read = time.monotonic()
        self._zero_read_count = 0
        print(f"[ReachyMicAudioInput] media opened reason={reason}", flush=True)
        self._logger.warning("ReachyMicAudioInput: media opened reason=%s", reason)

    async def _recover_media_if_needed(self, *, force: bool = False, reason: str = "periodic") -> None:
        if self._recovering_media or not self._daemon_url:
            return
        now = time.monotonic()
        if not force and now - self._last_media_check < self._media_check_interval:
            return
        self._last_media_check = now
        self._recovering_media = True
        try:
            try:
                status = await asyncio.to_thread(self._fetch_daemon_status, timeout=0.7)
            except Exception as status_error:
                self._logger.debug("ReachyMicAudioInput media status check failed: %s", status_error)
                return
            if not status.get("media_released"):
                if force:
                    self._logger.warning(
                        "ReachyMicAudioInput: forcing local media reopen reason=%s "
                        "daemon_state=%s daemon_error=%s media_released=%s",
                        reason,
                        status.get("state"),
                        status.get("error"),
                        status.get("media_released"),
                    )
                    print(
                        "[ReachyMicAudioInput] forcing local media reopen "
                        f"reason={reason} daemon_state={status.get('state')} "
                        f"daemon_error={status.get('error')}",
                        flush=True,
                    )
                    if await self._wait_for_producer(self._signalling_host, timeout=4):
                        self._open_media(reason=f"recover:{reason}:force")
                    return
                return
            self._logger.warning(
                "ReachyMicAudioInput: daemon media was released; reacquiring reason=%s",
                reason,
            )
            print(f"[ReachyMicAudioInput] daemon media released; reacquiring reason={reason}", flush=True)
            await asyncio.to_thread(self._acquire_daemon_media, timeout=2.0)
            signalling_host = str(status.get("wlan_ip") or self._signalling_host)
            self._signalling_host = signalling_host
            if not await self._wait_for_producer(signalling_host, timeout=4):
                self._logger.warning(
                    "ReachyMicAudioInput: WebRTC producer not ready after reacquire reason=%s",
                    reason,
                )
                return
            self._open_media(reason=f"recover:{reason}")
        finally:
            self._recovering_media = False

    async def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        if self._fallback is not None:
            return await self._fallback.read(rate=rate, duration=duration)
        if self._unavailable:
            if not self._warned_unavailable_read:
                print("[ReachyMicAudioInput] read silence: microphone source is unavailable", flush=True)
                self._warned_unavailable_read = True
            return np.zeros(int(self.rate * (duration or 0.1)), dtype=self.dtype)
        if not self._started or self._media is None:
            return np.zeros(int(self.rate * (duration or 0.1)), dtype=self.dtype)
        try:
            target_samples = int(self.rate * (duration or 0.1))
            await self._recover_media_if_needed(reason="read")
            # 预热期：前N次读取返回静音，避免 WebRTC 初始噪声误触发 VAD
            if self._warmup_reads > 0:
                self._warmup_reads -= 1
                # 消费掉音频数据但不返回（让 WebRTC buffer 排空）
                for _ in range(target_samples // 320 + 1):
                    self._media.get_audio_sample()
                    await asyncio.sleep(0.005)
                return np.zeros(target_samples, dtype=self.dtype)
            chunks = []
            collected = 0
            max_attempts = target_samples // 160 + 10
            for _ in range(max_attempts):
                sample = self._media.get_audio_sample()
                if sample is None:
                    await asyncio.sleep(0.01)
                    continue
                if sample.ndim > 1:
                    mono = sample[:, 0]
                else:
                    mono = sample
                if mono.dtype == np.float32:
                    mono = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
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
            rms = float(np.sqrt(np.mean(result.astype(float) ** 2)))
            if rms > 0.5:
                self._last_nonzero_read = time.monotonic()
                self._zero_read_count = 0
            else:
                self._zero_read_count += 1
                if (
                    self._zero_read_count >= self._zero_reopen_reads
                    and time.monotonic() - self._last_nonzero_read >= self._zero_reopen_seconds
                ):
                    await self._recover_media_if_needed(force=True, reason="zero_audio")
                if self._zero_fallback_reads > 0 and self._zero_read_count >= self._zero_fallback_reads:
                    if await self._switch_to_sounddevice_fallback(reason="persistent_zero_audio"):
                        return await self._fallback.read(rate=rate, duration=duration)
            if self._read_count <= 3 or self._read_count % 100 == 0:
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
        if self._media:
            try:
                self._media.close()
            except Exception:
                pass
            self._media = None
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
        self._last_restart_attempt = 0.0
        self._restart_attempts = 0
        self._restart_interval = float(os.environ.get("MOSS_SOUNDDEVICE_RESTART_INTERVAL_SECONDS", "1.0"))

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
            self._unavailable = False
            self._warned_unavailable_read = False
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
            self._restart_attempts = 0
        except Exception as e:
            self._stream = None
            self._unavailable = True
            print(f"[SoundDeviceAudioInput] unavailable: {e}", flush=True)
            self._logger.error("SoundDeviceAudioInput unavailable: %s", e)
            if self._raise_on_unavailable:
                raise

    async def _try_restart(self, reason: str) -> bool:
        if self._closed:
            return False
        now = time.time()
        if now - self._last_restart_attempt < self._restart_interval:
            return False
        self._last_restart_attempt = now
        self._restart_attempts += 1
        print(
            f"[SoundDeviceAudioInput] restarting input stream reason={reason} "
            f"attempt={self._restart_attempts}",
            flush=True,
        )
        self._logger.warning(
            "SoundDeviceAudioInput restarting input stream reason=%s attempt=%d",
            reason,
            self._restart_attempts,
        )
        try:
            await self.stop()
            with self._lock:
                self._buffer.clear()
            await self.start()
        except Exception as e:
            self._stream = None
            self._unavailable = True
            self._logger.error("SoundDeviceAudioInput restart failed: %s", e)
            print(f"[SoundDeviceAudioInput] restart failed: {e}", flush=True)
            return False
        return self._stream is not None and not self._unavailable

    async def read(self, *, rate=None, duration=None) -> np.ndarray:
        duration = duration or 0.1
        samples_needed = int(self.rate * duration)
        if self._unavailable or self._stream is None:
            restarted = await self._try_restart("unavailable" if self._unavailable else "no_stream")
            if restarted:
                self._warned_unavailable_read = False
            else:
                if not self._warned_unavailable_read:
                    print("[SoundDeviceAudioInput] read silence: local microphone is unavailable", flush=True)
                    self._warned_unavailable_read = True
                return np.zeros(samples_needed, dtype=self.dtype)
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
            try:
                self._stream.stop()
            except Exception as e:
                self._logger.debug("SoundDeviceAudioInput stop error: %s", e)
            try:
                self._stream.close()
            except Exception as e:
                self._logger.debug("SoundDeviceAudioInput close error: %s", e)
            self._stream = None

    async def close(self, error=None) -> None:
        if self._closed:
            return
        self._closed = True
        await self.stop()

    def closed(self) -> bool:
        return self._closed
