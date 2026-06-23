#!/usr/bin/env python3
"""Probe Reachy Mini robot microphone through the SDK WebRTC media path."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from typing import Any

import numpy as np


def _wait_for_reachymini_producer(host: str, timeout: float = 8.0) -> dict[str, Any]:
    from reachy_mini.media.webrtc_utils import get_producer_list

    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            producers = get_producer_list(host, 8443)
            if any(meta.get("name") == "reachymini" for meta in producers.values()):
                return {"ready": True, "producers": producers, "error": None}
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(0.25)
    return {"ready": False, "producers": {}, "error": last_error}


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _sample_stats(sample: Any) -> dict[str, Any] | None:
    if sample is None:
        return None
    arr = np.asarray(sample)
    if arr.size <= 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "rms": 0.0, "peak": 0.0}
    values = arr.astype(float)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "rms": float(np.sqrt(np.mean(values**2))),
        "peak": float(np.max(np.abs(values))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Reachy Mini robot microphone.")
    parser.add_argument("--host", default=os.environ.get("REACHY_ROBOT_HOST", "192.168.31.222"))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    os.environ["GST_PLUGIN_PATH"] = ""
    os.environ["GST_PLUGIN_SYSTEM_PATH"] = ""

    from reachy_mini import ReachyMini

    result: dict[str, Any] = {
        "host": args.host,
        "http_8000_open_before": _port_open(args.host, 8000),
        "webrtc_8443_open_before": _port_open(args.host, 8443),
        "connected": False,
        "samples": [],
        "error": None,
    }

    probe = None
    mini = None
    try:
        probe = ReachyMini(host=args.host, connection_mode="network", media_backend="no_media")
        result["probe_media_released"] = bool(getattr(probe, "media_released", False))
        if getattr(probe, "media_released", False):
            probe.acquire_media()
        result["webrtc_8443_open_after_acquire"] = _port_open(args.host, 8443)
        result["producer_wait"] = _wait_for_reachymini_producer(args.host)

        mini = ReachyMini(host=args.host, connection_mode="network", media_backend="default", log_level="WARNING")
        result["connected"] = True
        result["media_released"] = bool(getattr(mini, "media_released", False))
        result["input_samplerate"] = mini.media.get_input_audio_samplerate()
        result["output_samplerate"] = mini.media.get_output_audio_samplerate()

        for _ in range(args.samples):
            stats = _sample_stats(mini.media.get_audio_sample())
            if stats is not None:
                result["samples"].append(stats)
            time.sleep(args.sleep)

        result["sample_count"] = len(result["samples"])
        result["nonzero_sample_count"] = sum(1 for s in result["samples"] if float(s.get("peak", 0.0)) > 0)
        return 0 if result["sample_count"] > 0 else 2
    except Exception as exc:
        result["error"] = repr(exc)
        return 1
    finally:
        for obj in (mini, probe):
            if obj is None:
                continue
            try:
                obj.client.disconnect()
            except Exception:
                pass
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
