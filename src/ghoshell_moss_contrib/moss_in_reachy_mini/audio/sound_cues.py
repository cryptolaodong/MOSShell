"""Short Reachy Mini sound cues for voice-state feedback.

The cue layer is deliberately optional and side-channel only. It never replaces
TTS playback; it only schedules tiny non-speech WAV cues through the robot sound
API when enabled.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import threading
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from typing import Protocol

from .speaking_gate import is_speaking, is_speech_pending, mark_speaking_for


CueKind = str


def _truthy_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


class SoundCueTransport(Protocol):
    def play_wav(self, *, name: str, wav_bytes: bytes, timeout: float) -> dict[str, object]:
        """Upload and play one WAV cue."""


@dataclass(frozen=True)
class SoundCueResult:
    scheduled: bool
    kind: CueKind
    reason: str
    duration_seconds: float = 0.0


class HttpReachySoundCueTransport:
    def __init__(self, *, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def play_wav(self, *, name: str, wav_bytes: bytes, timeout: float) -> dict[str, object]:
        body, boundary = _multipart_file_body(filename=name, data=wav_bytes)
        req = urllib.request.Request(
            f"{self._base_url}/api/media/sounds/upload",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        robot_file = uploaded.get("path") or uploaded.get("file") or name
        played = _json_request(
            self._base_url,
            "/api/media/play_sound",
            method="POST",
            data={"file": robot_file},
            timeout=timeout,
        )
        return {"uploaded": uploaded, "played": played, "robot_file": robot_file}


class SoundCuePlayer:
    def __init__(
        self,
        transport: SoundCueTransport,
        *,
        enabled: bool = False,
        cooldown_seconds: float = 0.8,
        timeout_seconds: float = 3.0,
        gain: float = 0.35,
        suppress_when_speaking: bool = True,
        mark_output_gate: bool = True,
        logger=None,
    ) -> None:
        self._transport = transport
        self._enabled = bool(enabled)
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._timeout_seconds = max(0.2, float(timeout_seconds))
        self._gain = max(0.0, min(1.0, float(gain)))
        self._suppress_when_speaking = bool(suppress_when_speaking)
        self._mark_output_gate = bool(mark_output_gate)
        self._logger = logger
        self._lock = threading.Lock()
        self._last_played_at: dict[CueKind, float] = {}
        self._threads: list[threading.Thread] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def play(self, kind: CueKind, *, force: bool = False) -> SoundCueResult:
        kind = _normalize_kind(kind)
        if not self._enabled and not force:
            return SoundCueResult(False, kind, "disabled")
        if self._suppress_when_speaking and not force and (is_speaking(tail=0.15) or is_speech_pending()):
            return SoundCueResult(False, kind, "robot_output_active")
        now = time.monotonic()
        with self._lock:
            last = self._last_played_at.get(kind, 0.0)
            if not force and now - last < self._cooldown_seconds:
                return SoundCueResult(False, kind, "cooldown")
            self._last_played_at[kind] = now

        wav_bytes, duration = make_cue_wav(kind, gain=self._gain)
        if self._mark_output_gate:
            mark_speaking_for(duration, tail=0.35)
        name = f"moss_cue_{kind}_{int(time.time() * 1000)}.wav"
        thread = threading.Thread(
            target=self._play_worker,
            args=(kind, name, wav_bytes),
            name=f"ReachySoundCue-{kind}",
            daemon=True,
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return SoundCueResult(True, kind, "scheduled", duration)

    def wait_idle(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._lock:
                threads = [thread for thread in self._threads if thread.is_alive()]
                self._threads = threads
            if not threads:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            threads[0].join(timeout=min(0.05, remaining))

    def _play_worker(self, kind: CueKind, name: str, wav_bytes: bytes) -> None:
        try:
            result = self._transport.play_wav(
                name=name,
                wav_bytes=wav_bytes,
                timeout=self._timeout_seconds,
            )
            if self._logger:
                self._logger.info("[ReachySoundCue] kind=%s result=%s", kind, result)
        except Exception as error:
            if self._logger:
                self._logger.warning("[ReachySoundCue] kind=%s failed: %s", kind, error)


def build_reachy_sound_cue_player(*, logger=None) -> SoundCuePlayer:
    host = os.environ.get("REACHY_ROBOT_HOST", "reachy-mini.local").strip() or "reachy-mini.local"
    base_url = os.environ.get("MOSS_REACHY_SOUND_CUE_BASE_URL", f"http://{host}:8000")
    return SoundCuePlayer(
        HttpReachySoundCueTransport(base_url=base_url),
        enabled=_truthy_env("MOSS_REACHY_SOUND_CUES_ENABLED", False),
        cooldown_seconds=_env_float("MOSS_REACHY_SOUND_CUE_COOLDOWN_SECONDS", 0.8),
        timeout_seconds=_env_float("MOSS_REACHY_SOUND_CUE_TIMEOUT_SECONDS", 3.0, minimum=0.2),
        gain=_env_float("MOSS_REACHY_SOUND_CUE_GAIN", 0.35),
        suppress_when_speaking=_truthy_env("MOSS_REACHY_SOUND_CUE_SUPPRESS_WHEN_SPEAKING", True),
        mark_output_gate=_truthy_env("MOSS_REACHY_SOUND_CUE_MARK_SPEAKING", True),
        logger=logger,
    )


def make_cue_wav(kind: CueKind, *, sample_rate: int = 24000, gain: float = 0.35) -> tuple[bytes, float]:
    kind = _normalize_kind(kind)
    plans = {
        "listening": [(880.0, 0.055), (0.0, 0.025), (1175.0, 0.075)],
        "thinking": [(660.0, 0.045), (0.0, 0.025), (740.0, 0.045)],
        "done": [(1175.0, 0.06), (0.0, 0.025), (1568.0, 0.08)],
        "error": [(220.0, 0.11), (0.0, 0.035), (196.0, 0.13)],
        "recovery": [(392.0, 0.07), (0.0, 0.02), (784.0, 0.07)],
    }
    segments = plans[kind]
    amp = int(18000 * max(0.0, min(1.0, float(gain))))
    frames: list[bytes] = []
    total_samples = 0
    for hz, seconds in segments:
        count = int(sample_rate * seconds)
        for i in range(count):
            if hz <= 0:
                sample = 0
            else:
                fade = min(1.0, i / max(1, sample_rate // 160), (count - i) / max(1, sample_rate // 160))
                sample = int(amp * fade * math.sin(2 * math.pi * hz * i / sample_rate))
            frames.append(struct.pack("<h", sample))
        total_samples += count
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(frames))
    return buf.getvalue(), total_samples / sample_rate


def _normalize_kind(kind: CueKind) -> CueKind:
    normalized = (kind or "").strip().lower().replace("-", "_")
    aliases = {
        "start_listening": "listening",
        "listen": "listening",
        "think": "thinking",
        "complete": "done",
        "finished": "done",
        "fail": "error",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"listening", "thinking", "done", "error", "recovery"}:
        raise ValueError(f"unknown sound cue kind: {kind!r}")
    return normalized


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    data: dict[str, object] | None = None,
    timeout: float = 3.0,
) -> object:
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _multipart_file_body(*, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = f"moss-cue-{uuid.uuid4().hex}"
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
