#!/usr/bin/env python3
"""Play one WAV file through the Reachy Mini sound upload API.

This is a side-channel sound check for the new audio mixer/file-player work.
It does not start MOSS, does not touch ASR, and does not change the existing
upload-based TTS path.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> Any:
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _multipart_file_body(*, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = f"moss-sound-check-{uuid.uuid4().hex}"
    body = b"".join(
        [
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
    )
    return body, boundary


def _make_tone_wav(*, seconds: float = 1.2, rate: int = 24000, hz: float = 660.0) -> bytes:
    buf = io.BytesIO()
    frames = []
    total = int(seconds * rate)
    for i in range(total):
        fade = min(1.0, i / max(1, rate // 20), (total - i) / max(1, rate // 20))
        sample = int(18000 * fade * math.sin(2 * math.pi * hz * i / rate))
        frames.append(struct.pack("<h", sample))
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(frames))
    return buf.getvalue()


def _read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() not in {1, 2}:
            raise ValueError(f"unsupported WAV sample width: {wav.getsampwidth()}")
    return path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.31.222:8000")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if args.file:
        wav_bytes = _read_wav(args.file)
        filename = args.file.name
    else:
        wav_bytes = _make_tone_wav()
        filename = f"moss_sound_file_self_check_{int(time.time())}.wav"

    report: dict[str, Any] = {
        "base_url": base_url,
        "filename": filename,
        "bytes": len(wav_bytes),
        "checks": {},
    }
    failures: list[str] = []

    for name, path in {
        "daemon": "/api/daemon/status",
        "media": "/api/media/status",
        "volume": "/api/volume/current",
    }.items():
        try:
            report["checks"][name] = _json_request(base_url, path, timeout=args.timeout)
        except Exception as exc:
            report["checks"][name] = {"error": str(exc)}
            failures.append(name)

    try:
        body, boundary = _multipart_file_body(filename=filename, data=wav_bytes)
        req = urllib.request.Request(
            f"{base_url}/api/media/sounds/upload",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        robot_file = uploaded.get("path") or uploaded.get("file") or filename
        played = _json_request(
            base_url,
            "/api/media/play_sound",
            method="POST",
            data={"file": robot_file},
            timeout=args.timeout,
        )
        report["checks"]["uploaded"] = uploaded
        report["checks"]["played"] = played
        if played.get("status") != "ok":
            failures.append("play_sound")
    except Exception as exc:
        report["checks"]["play_error"] = {"error": str(exc)}
        failures.append("sound_file")

    report["passed"] = not failures
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
