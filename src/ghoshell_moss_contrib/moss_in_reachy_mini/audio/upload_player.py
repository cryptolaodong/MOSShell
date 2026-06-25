"""Upload-based audio player for Reachy Mini.

Bypasses WebRTC/GStreamer entirely by using the WebSocket upload protocol:
  1. upload_audio_start — declare an upload slot
  2. upload_audio_chunk — send base64 WAV chunks
  3. upload_audio_finish — finalize the upload
  4. play_uploaded_audio — tell daemon to play

This works on Mac (no GStreamer needed) and on any remote connection.
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import queue
import threading
import time
import wave
from typing import Optional
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.contracts.speech import AudioFormat, StreamAudioPlayer
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from reachy_mini import ReachyMini
from reachy_mini.io.protocol import (
    UploadAudioStartCmd,
    UploadAudioChunkCmd,
    UploadAudioFinishCmd,
    PlayUploadedAudioCmd,
    CancelAudioCmd,
)
from .speaking_gate import (
    clear_speaking,
    clear_speech_pending,
    latest_user_turn_at,
    mark_speech_pending,
    mark_speaking_for,
)

__all__ = ["ReachyMiniUploadAudioPlayer"]

MAX_CHUNK_SIZE = 15 * 1024  # Keep under 16KB limit per chunk


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _connect_without_releasing_media(**kwargs) -> ReachyMini:
    original_release_media = ReachyMini.release_media

    def _noop_release_media(self) -> None:
        self._media_released = True

    ReachyMini.release_media = _noop_release_media
    try:
        return ReachyMini(**kwargs)
    finally:
        ReachyMini.release_media = original_release_media


class ReachyMiniUploadAudioPlayer(StreamAudioPlayer):
    """Stream audio to Reachy Mini via WebSocket upload protocol.

    Collects PCM frames in a buffer, then when playback is requested,
    encodes as WAV, splits into base64 chunks, uploads via WebSocket,
    and tells the daemon to play.
    """

    def __init__(
        self,
        mini: ReachyMini,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float | None = None,
        min_play_seconds: float | None = None,
        max_buffer_wait: float | None = None,
        initial_min_play_seconds: float | None = None,
        initial_max_buffer_wait: float | None = None,
    ):
        self._mini = mini
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_type = AudioFormat.PCM_S16LE
        self.logger = logger or logging.getLogger("moss")
        self._safety_delay = (
            _env_float("MOSS_REACHY_UPLOAD_SAFETY_DELAY", 0.05)
            if safety_delay is None
            else max(0.0, safety_delay)
        )
        self._min_play_seconds = (
            _env_float("MOSS_REACHY_UPLOAD_MIN_PLAY_SECONDS", 1.4)
            if min_play_seconds is None
            else max(0.0, min_play_seconds)
        )
        self._max_buffer_wait = (
            _env_float("MOSS_REACHY_UPLOAD_MAX_BUFFER_WAIT", 0.65)
            if max_buffer_wait is None
            else max(0.0, max_buffer_wait)
        )
        self._initial_min_play_seconds = (
            _env_float("MOSS_REACHY_UPLOAD_INITIAL_MIN_PLAY_SECONDS", 0.65)
            if initial_min_play_seconds is None
            else max(0.0, initial_min_play_seconds)
        )
        self._initial_max_buffer_wait = (
            _env_float("MOSS_REACHY_UPLOAD_INITIAL_MAX_BUFFER_WAIT", 0.18)
            if initial_max_buffer_wait is None
            else max(0.0, initial_max_buffer_wait)
        )
        self._transport = os.environ.get("MOSS_REACHY_UPLOAD_TRANSPORT", "http").strip().lower()
        if self._transport not in {"http", "websocket", "auto"}:
            self._transport = "http"
        self._gain = _env_float("MOSS_REACHY_UPLOAD_GAIN", 1.0, minimum=0.1)
        self._http_timeout = _env_float("MOSS_REACHY_UPLOAD_HTTP_TIMEOUT", 5.0, minimum=0.2)
        self._health_log_interval = _env_float(
            "MOSS_REACHY_UPLOAD_HEALTH_LOG_INTERVAL",
            30.0,
            minimum=0.0,
        )
        self._last_health_log_at = 0.0
        self._log_prefix = "[ReachyMiniUploadAudioPlayer]"

        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._generation = 0
        self._play_done_event = ThreadSafeEvent()
        self._play_done_event.set()
        self._worker_idle_event = ThreadSafeEvent()
        self._worker_idle_event.set()
        self._estimated_end_time = 0.0
        self._next_play_monotonic = 0.0
        self._closed = False
        self._current_upload_id: Optional[str] = None
        self._played_segment_count = 0
        self._pending_started_at = 0.0
        self._pending_started_wall_at = 0.0
        self._pending_user_turn_at = 0.0
        self._first_audio_logged = False

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._work_queue: queue.Queue[Optional[tuple[list[np.ndarray], bool, int]]] = queue.Queue()

    async def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self.logger.info("%s started", self._log_prefix)

    async def close(self) -> None:
        self._closed = True
        self._stop_event.set()
        self._work_queue.put_nowait(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self.logger.info("%s closed", self._log_prefix)

    async def clear(self) -> None:
        with self._buffer_lock:
            self._audio_buffer.clear()
            self._generation += 1
            generation = self._generation
        self._drain_work_queue()
        self._work_queue.put_nowait(([], True, generation))
        self._worker_idle_event.set()
        clear_speaking()
        if self._current_upload_id:
            try:
                self._mini.client.send_command(CancelAudioCmd(upload_id=self._current_upload_id))
            except Exception:
                pass
            self._current_upload_id = None
        if self._transport in {"http", "auto"}:
            try:
                self._http_json("/api/media/stop_sound", method="POST", data={})
            except Exception:
                pass
        self._estimated_end_time = time.time()
        self._next_play_monotonic = time.monotonic()
        self._played_segment_count = 0
        self._pending_started_at = 0.0
        self._pending_started_wall_at = 0.0
        self._pending_user_turn_at = 0.0
        self._first_audio_logged = False
        self._play_done_event.set()
        self.logger.info("%s cleared", self._log_prefix)

    async def finish_stream(self) -> None:
        """Finish a normal stream without erasing the speech tail gate.

        ``clear`` is a hard interrupt and closes the speaking gate immediately.
        A naturally completed TTS stream should keep the tail window so body
        commands compiled right after a speech segment can still align with it.
        """
        with self._buffer_lock:
            self._audio_buffer.clear()
            self._generation += 1
        self._drain_work_queue()
        self._worker_idle_event.set()
        clear_speech_pending()
        self._play_done_event.set()
        self.logger.info("%s stream finished", self._log_prefix)

    def add(
        self,
        chunk: np.ndarray,
        *,
        audio_type: AudioFormat,
        rate: int,
        channels: int = 1,
    ) -> float:
        if self._closed:
            return time.time()

        if audio_type == AudioFormat.PCM_F32LE:
            audio_data = (chunk * 32767).astype(np.int16)
        else:
            audio_data = chunk.astype(np.int16)

        if rate != self.sample_rate:
            from scipy import signal as scipy_signal
            num_samples = int(len(audio_data) * float(self.sample_rate) / rate)
            audio_data = scipy_signal.resample(audio_data, num_samples).astype(np.int16)

        duration = len(audio_data) / self.sample_rate

        with self._buffer_lock:
            self._audio_buffer.append(audio_data)
        self._worker_idle_event.clear()

        if self._play_done_event.is_set():
            self._play_done_event.clear()
        if not self._first_audio_logged:
            self._first_audio_logged = True
            since_pending = (
                time.monotonic() - self._pending_started_at
                if self._pending_started_at
                else 0.0
            )
            self.logger.info(
                "%s [ReachyLatency] tts_first_audio_to_player duration=%.2fs since_pending=%.2fs",
                self._log_prefix,
                duration,
                since_pending,
            )

        current_time = time.time()
        if current_time > self._estimated_end_time:
            self._estimated_end_time = current_time + duration
        else:
            self._estimated_end_time += duration
        return self._estimated_end_time

    async def wait_play_done(self, timeout: Optional[float] = None) -> bool:
        # Flush current buffer to worker
        self._flush_buffer(force=True)

        while True:
            wall_remaining = (self._estimated_end_time + self._safety_delay) - time.time()
            play_remaining = self._next_play_monotonic - time.monotonic()
            time_to_wait = max(wall_remaining, play_remaining)
            if time_to_wait <= 0:
                break
            await asyncio.sleep(min(0.1, time_to_wait))

        while True:
            await self._play_done_event.wait()
            await self._worker_idle_event.wait()
            with self._buffer_lock:
                live_buffer_empty = not self._audio_buffer
            if live_buffer_empty and self._work_queue.empty():
                wall_remaining = (self._estimated_end_time + self._safety_delay) - time.time()
                play_remaining = self._next_play_monotonic - time.monotonic()
                time_to_wait = max(wall_remaining, play_remaining)
                if time_to_wait > 0.0:
                    await asyncio.sleep(min(0.1, time_to_wait))
                    continue
                break
            self._play_done_event.clear()
            self._worker_idle_event.clear()
            self._flush_buffer(force=True)

        self.logger.info("%s play done", self._log_prefix)
        return True

    def is_playing(self) -> bool:
        return time.time() < self._estimated_end_time or not self._play_done_event.is_set()

    def is_closed(self) -> bool:
        return self._closed

    def mark_pending(self) -> None:
        self._pending_started_at = time.monotonic()
        self._pending_started_wall_at = time.time()
        self._pending_user_turn_at = latest_user_turn_at()
        self._first_audio_logged = False
        mark_speech_pending()
        user_turn_age = (
            self._pending_started_wall_at - self._pending_user_turn_at
            if self._pending_user_turn_at
            else 0.0
        )
        self.logger.info("%s speech pending user_turn_age=%.2fs", self._log_prefix, user_turn_age)

    def on_play(self, callback) -> None:
        pass

    def on_play_done(self, callback) -> None:
        pass

    def _flush_buffer(self, *, force: bool = False) -> None:
        with self._buffer_lock:
            if not self._audio_buffer and not force:
                return
            frames = self._audio_buffer.copy()
            self._audio_buffer.clear()
            generation = self._generation
        if frames or force:
            self._worker_idle_event.clear()
        self._work_queue.put_nowait((frames, force, generation))

    def _drain_work_queue(self) -> None:
        while True:
            try:
                self._work_queue.get_nowait()
            except queue.Empty:
                return

    def _buffered_duration(self, frames: list[np.ndarray]) -> float:
        if not frames:
            return 0.0
        return sum(len(frame) for frame in frames) / float(self.sample_rate)

    def _stale_response_status(self) -> tuple[bool, str, dict[str, float]]:
        if not _env_enabled("MOSS_REACHY_UPLOAD_DROP_STALE_RESPONSE", "1"):
            return False, "", {}
        now = time.time()
        latest_turn = latest_user_turn_at()
        max_age = _env_float("MOSS_REACHY_UPLOAD_MAX_RESPONSE_AGE_SECONDS", 5.0)
        new_turn_grace = _env_float("MOSS_REACHY_UPLOAD_NEW_TURN_GRACE_SECONDS", 0.15)
        tracking_window = _env_float("MOSS_REACHY_UPLOAD_STALE_TRACKING_WINDOW_SECONDS", 20.0)
        pending_wall = self._pending_started_wall_at or now
        pending_turn = self._pending_user_turn_at
        if pending_turn and tracking_window > 0 and pending_wall - pending_turn > tracking_window:
            return False, "", {
                "latest_user_turn_at": round(latest_turn, 6),
                "pending_started_wall_at": round(pending_wall, 6),
                "pending_user_turn_at": round(pending_turn, 6),
                "tracking_window": round(tracking_window, 3),
            }
        fields = {
            "latest_user_turn_at": round(latest_turn, 6),
            "pending_started_wall_at": round(pending_wall, 6),
            "pending_user_turn_at": round(pending_turn, 6),
            "age_since_pending_turn": round(now - pending_turn, 3) if pending_turn else 0.0,
            "latest_after_pending": round(latest_turn - pending_wall, 3) if latest_turn else 0.0,
            "max_age": round(max_age, 3),
            "tracking_window": round(tracking_window, 3),
        }
        if latest_turn and latest_turn > pending_wall + new_turn_grace:
            return True, "new_user_turn_after_tts_pending", fields
        if pending_turn and max_age > 0 and now - pending_turn > max_age:
            return True, "response_too_old_for_user_turn", fields
        return False, "", fields

    def _worker_loop(self) -> None:
        pending_frames: list[np.ndarray] = []
        last_buffered_at = 0.0
        pending_generation = self._generation

        def drain_live_buffer() -> None:
            nonlocal last_buffered_at, pending_generation, pending_frames
            with self._buffer_lock:
                if not self._audio_buffer:
                    return
                if pending_generation != self._generation:
                    pending_frames = []
                    pending_generation = self._generation
                pending_frames.extend(self._audio_buffer)
                self._audio_buffer.clear()
                self._worker_idle_event.clear()
                last_buffered_at = time.time()

        def mark_idle_if_drained() -> None:
            with self._buffer_lock:
                live_buffer_empty = not self._audio_buffer
            if not pending_frames and live_buffer_empty and self._work_queue.empty():
                self._worker_idle_event.set()

        def maybe_play(*, force: bool = False) -> None:
            nonlocal pending_frames, last_buffered_at, pending_generation
            if not pending_frames:
                mark_idle_if_drained()
                return
            duration = self._buffered_duration(pending_frames)
            waited = time.time() - last_buffered_at if last_buffered_at else 0.0
            is_first_segment = self._played_segment_count == 0
            min_play_seconds = (
                self._initial_min_play_seconds if is_first_segment else self._min_play_seconds
            )
            max_buffer_wait = (
                self._initial_max_buffer_wait if is_first_segment else self._max_buffer_wait
            )
            if not force and duration < min_play_seconds and waited < max_buffer_wait:
                return
            frames = pending_frames
            generation = pending_generation
            pending_frames = []
            last_buffered_at = 0.0
            self._upload_and_play(frames, generation)
            mark_idle_if_drained()

        while not self._stop_event.is_set():
            try:
                item = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                drain_live_buffer()
                maybe_play(force=False)
                continue

            if item is None:
                break

            frames, force, generation = item
            if generation != pending_generation:
                pending_frames = []
                pending_generation = generation
            if frames:
                pending_frames.extend(frames)
                self._worker_idle_event.clear()
                last_buffered_at = time.time()

            while not self._work_queue.empty():
                try:
                    more = self._work_queue.get_nowait()
                    if more is None:
                        break
                    more_frames, more_force, more_generation = more
                    if more_generation != pending_generation:
                        pending_frames = []
                        pending_generation = more_generation
                    if more_frames:
                        pending_frames.extend(more_frames)
                        self._worker_idle_event.clear()
                        last_buffered_at = time.time()
                    force = force or more_force
                except queue.Empty:
                    break

            drain_live_buffer()
            maybe_play(force=force)

        maybe_play(force=True)
        self._play_done_event.set()
        self._worker_idle_event.set()

    def _upload_and_play(self, frames: list[np.ndarray], generation: int) -> None:
        upload_id: str | None = None
        try:
            upload_started_at = time.monotonic()
            if generation != self._generation:
                return
            if not frames:
                self._play_done_event.set()
                return

            all_audio = np.concatenate(frames)
            if len(all_audio) == 0:
                self._play_done_event.set()
                return

            stale, stale_reason, stale_fields = self._stale_response_status()
            if stale:
                clear_speech_pending()
                self.logger.warning(
                    "%s [ReachyLatency] tts_stale_drop reason=%s generation=%d fields=%s",
                    self._log_prefix,
                    stale_reason,
                    generation,
                    stale_fields,
                )
                self._play_done_event.set()
                return

            all_audio = self._apply_gain(all_audio)

            # Encode as WAV
            wav_bytes = self._encode_wav(all_audio)
            b64_data = base64.b64encode(wav_bytes).decode('ascii')

            # Split into chunks
            total_chunks = math.ceil(len(b64_data) / MAX_CHUNK_SIZE)
            if total_chunks == 0:
                self._play_done_event.set()
                return

            duration = len(all_audio) / self.sample_rate
            transport_used = "unknown"
            play_result: dict[str, object] = {}
            for attempt in range(2):
                if generation != self._generation:
                    return
                upload_id = str(uuid4())
                self._current_upload_id = upload_id
                try:
                    while time.monotonic() < self._next_play_monotonic:
                        if self._stop_event.is_set() or generation != self._generation:
                            return
                        time.sleep(min(0.05, self._next_play_monotonic - time.monotonic()))
                    if generation != self._generation:
                        return
                    stale, stale_reason, stale_fields = self._stale_response_status()
                    if stale:
                        clear_speech_pending()
                        self.logger.warning(
                            "%s [ReachyLatency] tts_stale_drop_before_play reason=%s generation=%d fields=%s",
                            self._log_prefix,
                            stale_reason,
                            generation,
                            stale_fields,
                        )
                        self._play_done_event.set()
                        return
                    self._ensure_motors_ready(reason="before_play")
                    self._log_output_health(reason="before_play")
                    if self._transport in {"http", "auto"}:
                        try:
                            play_result = self._http_upload_and_play(
                                upload_id=upload_id,
                                wav_bytes=wav_bytes,
                            )
                            transport_used = "http"
                            break
                        except Exception as e:
                            self.logger.warning(
                                "%s HTTP upload/play failed: %s; trying websocket transport",
                                self._log_prefix,
                                e,
                            )
                            if self._transport == "http" and attempt >= 1:
                                raise
                    self._send_upload_commands(
                        upload_id=upload_id,
                        total_chunks=total_chunks,
                        b64_data=b64_data,
                        generation=generation,
                    )
                    self._mini.client.send_command(PlayUploadedAudioCmd(upload_id=upload_id))
                    transport_used = "websocket"
                    play_result = {"status": "sent"}
                    break
                except ConnectionError:
                    if attempt >= 1:
                        raise
                    self.logger.warning(
                        "%s upload/play connection lost; reconnecting and retrying once",
                        self._log_prefix,
                    )
                    self._reconnect_robot_client()
            self._played_segment_count += 1
            play_until_monotonic = time.monotonic() + duration + self._safety_delay
            self._next_play_monotonic = play_until_monotonic
            speaking_until_estimate = time.time() + duration + self._safety_delay
            self._estimated_end_time = max(self._estimated_end_time, speaking_until_estimate)
            pending_to_play = (
                time.monotonic() - self._pending_started_at
                if self._pending_started_at
                else 0.0
            )
            self.logger.info(
                "%s uploaded %d bytes (%.1fs audio), playing upload_id=%s transport=%s play_result=%s speaking_until=%.3f upload_elapsed=%.2fs pending_to_play=%.2fs chunks=%d gain=%.2f",
                self._log_prefix,
                len(wav_bytes),
                duration,
                upload_id,
                transport_used,
                play_result,
                speaking_until_estimate,
                time.monotonic() - upload_started_at,
                pending_to_play,
                total_chunks,
                self._gain,
            )
            mark_speaking_for(duration + self._safety_delay)

        except Exception as e:
            self.logger.exception("%s upload/play failed: %s", self._log_prefix, e)
        finally:
            if generation == self._generation:
                self._play_done_event.set()

    def _send_upload_commands(
        self,
        *,
        upload_id: str,
        total_chunks: int,
        b64_data: str,
        generation: int,
    ) -> None:
        self._mini.client.send_command(UploadAudioStartCmd(
            upload_id=upload_id,
            total_chunks=total_chunks,
            encoding="wav-base64",
        ))
        for i in range(total_chunks):
            if self._stop_event.is_set() or generation != self._generation:
                return
            start = i * MAX_CHUNK_SIZE
            end = start + MAX_CHUNK_SIZE
            chunk_data = b64_data[start:end]
            self._mini.client.send_command(UploadAudioChunkCmd(
                upload_id=upload_id,
                chunk_index=i,
                chunk=chunk_data,
            ))
        self._mini.client.send_command(UploadAudioFinishCmd(
            upload_id=upload_id,
        ))

    def _reconnect_robot_client(self) -> None:
        try:
            self._mini.client.disconnect()
        except Exception:
            pass
        host = getattr(self._mini, "host", os.environ.get("REACHY_ROBOT_HOST", "reachy-mini.local"))
        port = int(getattr(self._mini, "port", os.environ.get("REACHY_ROBOT_PORT", "8000")))
        connection_mode = getattr(self._mini, "connection_mode", "network")
        self._mini = _connect_without_releasing_media(
            host=host,
            port=port,
            connection_mode=connection_mode,
            media_backend="no_media",
            log_level="WARNING",
        )
        self.logger.info("%s robot client reconnected", self._log_prefix)

    def _apply_gain(self, audio_data: np.ndarray) -> np.ndarray:
        if self._gain == 1.0 or len(audio_data) == 0:
            return audio_data
        boosted = np.clip(audio_data.astype(np.float32) * self._gain, -32768, 32767)
        return boosted.astype(np.int16)

    def _robot_base_url(self) -> str:
        host = str(getattr(self._mini, "host", os.environ.get("REACHY_ROBOT_HOST", "reachy-mini.local")))
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        port = int(getattr(self._mini, "port", os.environ.get("REACHY_ROBOT_PORT", "8000")))
        return f"http://{host}:{port}"

    def _http_json(
        self,
        path: str,
        *,
        method: str = "GET",
        data: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> dict[str, object]:
        body = None
        headers = {}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(
            f"{self._robot_base_url()}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        with urlopen(req, timeout=timeout or self._http_timeout) as response:
            raw = response.read().decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def _multipart_file_body(self, *, filename: str, data: bytes) -> tuple[bytes, str]:
        boundary = f"moss-{uuid4().hex}"
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        return b"".join(parts), boundary

    def _http_upload_and_play(self, *, upload_id: str, wav_bytes: bytes) -> dict[str, object]:
        filename = f"moss_{upload_id}.wav"
        body, boundary = self._multipart_file_body(filename=filename, data=wav_bytes)
        req = Request(
            f"{self._robot_base_url()}/api/media/sounds/upload",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urlopen(req, timeout=max(self._http_timeout, 8.0)) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        robot_file = uploaded.get("path") or uploaded.get("file") or filename
        played = self._http_json(
            "/api/media/play_sound",
            method="POST",
            data={"file": str(robot_file)},
            timeout=max(self._http_timeout, 8.0),
        )
        return {
            "status": played.get("status", "unknown"),
            "file": str(robot_file),
            "uploaded": uploaded,
            "played": played,
        }

    def _ensure_motors_ready(self, *, reason: str) -> None:
        if not _env_enabled("MOSS_REACHY_AUTO_ENABLE_MOTORS", "1"):
            return
        try:
            motors = self._http_json("/api/motors/status", timeout=1.0)
            if str(motors.get("mode", "")).lower() == "enabled":
                return
            changed = self._http_json(
                "/api/motors/set_mode/enabled",
                method="POST",
                data={},
                timeout=2.0,
            )
            time.sleep(0.1)
            after = self._http_json("/api/motors/status", timeout=1.0)
            self.logger.warning(
                "%s auto_enabled_motors reason=%s before=%s changed=%s after=%s",
                self._log_prefix,
                reason,
                motors,
                changed,
                after,
            )
        except Exception as e:
            self.logger.warning("%s auto_enable_motors failed reason=%s: %s", self._log_prefix, reason, e)

    def _log_output_health(self, *, reason: str) -> None:
        if self._health_log_interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_health_log_at < self._health_log_interval:
            return
        self._last_health_log_at = now
        try:
            daemon = self._http_json("/api/daemon/status", timeout=1.0)
            media = self._http_json("/api/media/status", timeout=1.0)
            volume = self._http_json("/api/volume/current", timeout=1.0)
            motors = self._http_json("/api/motors/status", timeout=1.0)
            self.logger.info(
                "%s output_health reason=%s daemon_state=%s media=%s volume=%s motors=%s",
                self._log_prefix,
                reason,
                daemon.get("state"),
                media,
                volume,
                motors,
            )
        except Exception as e:
            self.logger.warning("%s output_health failed: %s", self._log_prefix, e)

    def _encode_wav(self, audio_data: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()
