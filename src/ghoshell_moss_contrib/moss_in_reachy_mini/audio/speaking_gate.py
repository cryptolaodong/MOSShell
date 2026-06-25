"""Small file-based speaking gate shared by Ghost and voice sensor processes."""

from __future__ import annotations

import os
import time
from pathlib import Path

TAIL_SECONDS = 1.2
PENDING_SECONDS = 30.0
THINKING_SECONDS = 12.0


def _gate_path() -> Path:
    workspace = Path(os.environ.get("MOSS_WORKSPACE", "/Users/laodong/Documents/MOSShell/.moss_ws"))
    state_dir = workspace / "runtime" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "reachy_speaking_until"


def _pending_path() -> Path:
    workspace = Path(os.environ.get("MOSS_WORKSPACE", "/Users/laodong/Documents/MOSShell/.moss_ws"))
    state_dir = workspace / "runtime" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "reachy_speech_pending_until"


def _thinking_path() -> Path:
    workspace = Path(os.environ.get("MOSS_WORKSPACE", "/Users/laodong/Documents/MOSShell/.moss_ws"))
    state_dir = workspace / "runtime" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "reachy_thinking_until"


def _user_turn_path() -> Path:
    workspace = Path(os.environ.get("MOSS_WORKSPACE", "/Users/laodong/Documents/MOSShell/.moss_ws"))
    state_dir = workspace / "runtime" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "reachy_last_user_turn_at"


def _write_timestamp(path: Path, value: float) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(f"{value:.6f}\n", encoding="utf-8")
    tmp.replace(path)


def mark_speaking_for(duration: float, *, tail: float = TAIL_SECONDS) -> float:
    """Mark robot output as active for duration + tail seconds."""
    clear_speech_pending()
    clear_thinking()
    now = time.time()
    until = now + max(0.0, duration) + max(0.0, tail)
    path = _gate_path()
    current = _read_until(path)
    if current is not None and current > until:
        until = current
    _write_timestamp(path, until)
    return until


def clear_speaking() -> float:
    """Close the speaking gate immediately."""
    clear_speech_pending()
    clear_thinking()
    until = time.time()
    path = _gate_path()
    _write_timestamp(path, until)
    return until


def is_speaking(*, tail: float = 0.0) -> bool:
    until = _read_until(_gate_path())
    if until is None:
        return False
    return time.time() <= until + max(0.0, tail)


def mark_speech_pending(timeout: float = PENDING_SECONDS) -> float:
    """Mark that TTS is active but uploaded playback has not started yet."""
    clear_thinking()
    until = time.time() + max(0.0, timeout)
    path = _pending_path()
    _write_timestamp(path, until)
    return until


def clear_speech_pending() -> float:
    until = time.time()
    path = _pending_path()
    _write_timestamp(path, until)
    return until


def is_speech_pending() -> bool:
    until = _read_until(_pending_path())
    if until is None:
        return False
    return time.time() <= until


def mark_thinking(timeout: float = THINKING_SECONDS) -> float:
    """Mark that Reachy is waiting for the next response to start."""
    until = time.time() + max(0.0, timeout)
    path = _thinking_path()
    _write_timestamp(path, until)
    return until


def clear_thinking() -> float:
    until = time.time()
    path = _thinking_path()
    _write_timestamp(path, until)
    return until


def mark_user_turn(timestamp: float | None = None) -> float:
    """Record when a user turn was accepted into MOSS."""
    ts = time.time() if timestamp is None else float(timestamp)
    _write_timestamp(_user_turn_path(), ts)
    return ts


def latest_user_turn_at() -> float:
    """Return the last accepted user turn timestamp, or 0 when unknown."""
    return _read_until(_user_turn_path()) or 0.0


def is_thinking() -> bool:
    until = _read_until(_thinking_path())
    if until is None:
        return False
    return time.time() <= until


def remaining_seconds() -> float:
    until = _read_until(_gate_path())
    if until is None:
        return 0.0
    return max(0.0, until - time.time())


def _read_until(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
