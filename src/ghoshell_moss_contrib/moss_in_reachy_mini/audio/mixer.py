"""Small audio mixer primitives for Reachy Mini side-channel sounds.

This module is intentionally independent from the current upload-based TTS
player. It gives prompts, sound effects, and file playback one controlled
output layer without changing the speech path.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PcmAudioSink(Protocol):
    def push_pcm(self, pcm: np.ndarray, *, sample_rate: int, channels: int) -> None:
        """Push one int16 PCM chunk to the output device."""

    def stop(self) -> None:
        """Best-effort output stop hook."""


@dataclass(frozen=True)
class AudioClip:
    name: str
    pcm: np.ndarray
    sample_rate: int
    channels: int = 1

    @property
    def frame_count(self) -> int:
        return int(_as_frames(self.pcm, self.channels).shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / max(1, int(self.sample_rate))


@dataclass(frozen=True)
class AudioPlaybackStatus:
    state: str
    current: str | None
    queued: int
    volume: float
    position_seconds: float
    duration_seconds: float


def _as_frames(pcm: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(pcm)
    if arr.size == 0:
        return np.zeros((0, max(1, int(channels))), dtype=np.int16)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -32768, 32767).astype(np.int16)
    ch = max(1, int(channels))
    if arr.ndim == 1:
        usable = (arr.size // ch) * ch
        arr = arr[:usable].reshape((-1, ch))
    elif arr.ndim == 2:
        if arr.shape[1] != ch and arr.shape[0] == ch:
            arr = arr.T
        if arr.shape[1] != ch:
            if ch == 1:
                arr = arr.mean(axis=1, dtype=np.float32).astype(np.int16).reshape((-1, 1))
            elif arr.shape[1] == 1:
                arr = np.repeat(arr, ch, axis=1)
            else:
                arr = arr[:, :ch]
    else:
        raise ValueError("PCM must be a 1D or 2D numpy array")
    return np.ascontiguousarray(arr, dtype=np.int16)


def _apply_volume(chunk: np.ndarray, volume: float) -> np.ndarray:
    if volume == 1.0:
        return chunk
    scaled = chunk.astype(np.float32) * float(volume)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


class AudioMixer:
    """Queue-based PCM player for non-TTS sounds.

    The mixer owns a single worker thread. Producers enqueue complete clips;
    the worker chunks them, applies master volume, and sends them to a sink.
    """

    def __init__(
        self,
        sink: PcmAudioSink,
        *,
        chunk_ms: int = 20,
        realtime: bool = True,
    ) -> None:
        self._sink = sink
        self._chunk_ms = max(1, int(chunk_ms))
        self._realtime = bool(realtime)
        self._queue: queue.Queue[AudioClip | None] = queue.Queue()
        self._lock = threading.RLock()
        self._idle = threading.Event()
        self._idle.set()
        self._pause = threading.Event()
        self._pause.clear()
        self._stop_current = threading.Event()
        self._closed = False
        self._volume = 1.0
        self._state = "idle"
        self._current: AudioClip | None = None
        self._position_frames = 0
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._closed = False
            self._worker = threading.Thread(target=self._worker_loop, name="ReachyAudioMixer", daemon=True)
            self._worker.start()

    def close(self, *, timeout: float = 1.0) -> None:
        with self._lock:
            self._closed = True
        self._pause.clear()
        self._stop_current.set()
        self._drain_queue()
        self._queue.put_nowait(None)
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=max(0.0, float(timeout)))
        try:
            self._sink.stop()
        except Exception:
            pass

    def play_clip(self, clip: AudioClip, *, clear_queue: bool = False) -> None:
        self.start()
        if clear_queue:
            self.clear()
        with self._lock:
            self._state = "queued"
            self._idle.clear()
        self._queue.put_nowait(clip)

    def clear(self) -> None:
        self._drain_queue()
        self._stop_current.set()
        self._pause.clear()
        try:
            self._sink.stop()
        except Exception:
            pass
        with self._lock:
            self._current = None
            self._position_frames = 0
            self._state = "stopped"
        self._idle.set()

    def stop(self) -> None:
        self.clear()

    def pause(self) -> None:
        with self._lock:
            if self._state in {"playing", "queued"}:
                self._state = "paused"
                self._pause.set()

    def resume(self) -> None:
        with self._lock:
            if self._state == "paused":
                self._state = "playing" if self._current is not None else "queued"
        self._pause.clear()

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, float(volume)))

    def status(self) -> AudioPlaybackStatus:
        with self._lock:
            current = self._current
            sample_rate = max(1, int(current.sample_rate)) if current else 1
            return AudioPlaybackStatus(
                state=self._state,
                current=current.name if current else None,
                queued=self._queue.qsize(),
                volume=self._volume,
                position_seconds=self._position_frames / sample_rate,
                duration_seconds=current.duration_seconds if current else 0.0,
            )

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout=timeout)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _worker_loop(self) -> None:
        while True:
            clip = self._queue.get()
            if clip is None:
                self._queue.task_done()
                return
            self._play_one(clip)
            self._queue.task_done()
            with self._lock:
                if self._queue.empty() and self._current is None and self._state != "stopped":
                    self._state = "idle"
                    self._idle.set()

    def _play_one(self, clip: AudioClip) -> None:
        frames = _as_frames(clip.pcm, clip.channels)
        chunk_frames = max(1, int(clip.sample_rate * self._chunk_ms / 1000))
        self._stop_current.clear()
        with self._lock:
            self._current = clip
            self._position_frames = 0
            self._state = "playing"

        for start in range(0, len(frames), chunk_frames):
            if self._stop_current.is_set():
                break
            while self._pause.is_set() and not self._stop_current.is_set():
                time.sleep(0.005)
            if self._stop_current.is_set():
                break

            end = min(len(frames), start + chunk_frames)
            with self._lock:
                volume = self._volume
            chunk = _apply_volume(frames[start:end], volume)
            self._sink.push_pcm(chunk, sample_rate=clip.sample_rate, channels=clip.channels)
            with self._lock:
                self._position_frames = end
            if self._realtime:
                time.sleep((end - start) / max(1, int(clip.sample_rate)))

        with self._lock:
            self._current = None
            self._position_frames = 0
            if self._state == "playing":
                self._state = "idle"
            if self._queue.empty():
                self._idle.set()
