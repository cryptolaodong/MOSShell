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
import logging
import math
import queue
import threading
import time
import wave
from typing import Optional
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

__all__ = ["ReachyMiniUploadAudioPlayer"]

MAX_CHUNK_SIZE = 15 * 1024  # Keep under 16KB limit per chunk


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
        safety_delay: float = 0.3,
    ):
        self._mini = mini
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_type = AudioFormat.PCM_S16LE
        self.logger = logger or logging.getLogger("moss")
        self._safety_delay = safety_delay
        self._log_prefix = "[ReachyMiniUploadAudioPlayer]"

        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._play_done_event = ThreadSafeEvent()
        self._play_done_event.set()
        self._estimated_end_time = 0.0
        self._closed = False
        self._current_upload_id: Optional[str] = None

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._work_queue: queue.Queue[Optional[list[np.ndarray]]] = queue.Queue()

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
        if self._current_upload_id:
            try:
                self._mini.client.send_command(CancelAudioCmd(upload_id=self._current_upload_id))
            except Exception:
                pass
            self._current_upload_id = None
        self._estimated_end_time = time.time()
        self._play_done_event.set()
        self.logger.info("%s cleared", self._log_prefix)

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

        if self._play_done_event.is_set():
            self._play_done_event.clear()

        current_time = time.time()
        if current_time > self._estimated_end_time:
            self._estimated_end_time = current_time + duration
        else:
            self._estimated_end_time += duration
        return self._estimated_end_time

    async def wait_play_done(self, timeout: Optional[float] = None) -> bool:
        # Flush current buffer to worker
        self._flush_buffer()

        time_to_wait = (self._estimated_end_time + self._safety_delay) - time.time()
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)

        await self._play_done_event.wait()
        self.logger.info("%s play done", self._log_prefix)
        return True

    def is_playing(self) -> bool:
        return time.time() < self._estimated_end_time or not self._play_done_event.is_set()

    def is_closed(self) -> bool:
        return self._closed

    def on_play(self, callback) -> None:
        pass

    def on_play_done(self, callback) -> None:
        pass

    def _flush_buffer(self) -> None:
        with self._buffer_lock:
            if not self._audio_buffer:
                return
            frames = self._audio_buffer.copy()
            self._audio_buffer.clear()
        self._work_queue.put_nowait(frames)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._work_queue.get(timeout=0.3)
            except queue.Empty:
                # Check if there's buffered data to flush
                with self._buffer_lock:
                    if self._audio_buffer:
                        frames = self._audio_buffer.copy()
                        self._audio_buffer.clear()
                    else:
                        frames = None
                if frames:
                    self._upload_and_play(frames)
                continue

            if item is None:
                break

            # Collect more frames that arrive quickly (batch them up)
            all_frames = list(item)
            time.sleep(0.6)  # Wait for more TTS data to arrive
            # Drain the queue and buffer
            while not self._work_queue.empty():
                try:
                    more = self._work_queue.get_nowait()
                    if more is None:
                        break
                    all_frames.extend(more)
                except queue.Empty:
                    break
            with self._buffer_lock:
                if self._audio_buffer:
                    all_frames.extend(self._audio_buffer)
                    self._audio_buffer.clear()

            self._upload_and_play(all_frames)

        self._play_done_event.set()

    def _upload_and_play(self, frames: list[np.ndarray]) -> None:
        try:
            if not frames:
                self._play_done_event.set()
                return

            all_audio = np.concatenate(frames)
            if len(all_audio) == 0:
                self._play_done_event.set()
                return

            # Encode as WAV
            wav_bytes = self._encode_wav(all_audio)
            b64_data = base64.b64encode(wav_bytes).decode('ascii')

            # Split into chunks
            total_chunks = math.ceil(len(b64_data) / MAX_CHUNK_SIZE)
            if total_chunks == 0:
                self._play_done_event.set()
                return

            upload_id = str(uuid4())
            self._current_upload_id = upload_id

            # 1. Start upload
            self._mini.client.send_command(UploadAudioStartCmd(
                upload_id=upload_id,
                total_chunks=total_chunks,
                encoding="wav-base64",
            ))

            # 2. Send chunks
            for i in range(total_chunks):
                start = i * MAX_CHUNK_SIZE
                end = start + MAX_CHUNK_SIZE
                chunk_data = b64_data[start:end]
                self._mini.client.send_command(UploadAudioChunkCmd(
                    upload_id=upload_id,
                    chunk_index=i,
                    chunk=chunk_data,
                ))

            # 3. Finish upload
            self._mini.client.send_command(UploadAudioFinishCmd(
                upload_id=upload_id,
            ))

            # 4. Play
            self._mini.client.send_command(PlayUploadedAudioCmd(
                upload_id=upload_id,
            ))

            # Wait for estimated playback duration
            duration = len(all_audio) / self.sample_rate
            self.logger.info(
                "%s uploaded %d bytes (%.1fs audio), playing upload_id=%s",
                self._log_prefix, len(wav_bytes), duration, upload_id,
            )
            time.sleep(duration + self._safety_delay)

        except Exception as e:
            self.logger.exception("%s upload/play failed: %s", self._log_prefix, e)
        finally:
            self._current_upload_id = None
            self._play_done_event.set()

    def _encode_wav(self, audio_data: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()
