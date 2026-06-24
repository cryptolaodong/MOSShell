#!/usr/bin/env python3
"""Smoke-test the live MOSS text input loop through the Zenoh signal bus."""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ghoshell_moss.core.blueprint.mindflow import InputSignal
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh


DEFAULT_MESSAGES = [
    "烟测1：只用一句中文回答“收到一”，不要调用任何机器人动作。",
    "烟测2：只用一句中文回答“收到二”，不要调用任何机器人动作。",
    "烟测3：先说“动作同步测试”，同时只调用一次开心表情，别调用其他动作。",
    "烟测4：只用一句中文回答“收到四”，不要调用任何机器人动作。",
    "烟测5：先说“点头同步测试”，同时只调用一次 head_move，duration=0.3，别调用其他动作。",
]

ATTENTION_RE = re.compile(r"set attention <Attention id=([^>\s]+)>")
SETTLED_RE = re.compile(r"interpreter settled:")
TTS_STREAM_RE = re.compile(r"Starting playing TTS stream")
PLAY_RE = re.compile(r"\[ReachyMiniUploadAudioPlayer\] uploaded .* playing upload_id=")
PLAY_DONE_RE = re.compile(r"\[ReachyMiniUploadAudioPlayer\] play done")
BASE_TTS_CLEAR_RE = re.compile(r"\[BaseTTSSpeech\] clear")
PLAYER_CLEAR_RE = re.compile(r"\[ReachyMiniUploadAudioPlayer\] cleared")
BODY_COMMAND_RE = re.compile(r"send command task apps\.bodies_reachymini:(\w+)")
SPEECH_SYNC_RE = re.compile(r"\[ReachyMiniSpeechSync\] command=(\w+) speech_started")
SPEECH_TIMEOUT_RE = re.compile(r"\[ReachyMiniSpeechSync\] command=(\w+) speech_start_timeout")
SYNC_REQUIRED_COMMANDS = {
    "dance",
    "emotion",
    "head_move",
    "head_reset",
    "antennas_move",
    "antennas_reset",
}


@dataclass
class WindowResult:
    idx: int
    message: str
    attention_ids: list[str]
    settled_count: int
    tts_stream_count: int
    play_count: int
    body_commands: list[str]
    sync_commands: list[str]
    failures: list[str]


def read_from(log_path: Path, offset: int) -> str:
    if not log_path.exists():
        return ""
    if log_path.stat().st_size < offset:
        offset = 0
    with log_path.open("rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def publish_input(z: "zenoh.Session", scope: str, text: str, description: str) -> None:
    signal = InputSignal().to_signal(text, description=description, stale_timeout=90.0)
    z.put(f"MOSS/{scope}/signals", signal.to_json())
    time.sleep(0.5)


def wait_for_settled(log_path: Path, offset: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = read_from(log_path, offset)
        if SETTLED_RE.search(text):
            return text
        time.sleep(0.25)
    return read_from(log_path, offset)


def collect_repeat_window(log_path: Path, offset: int, seconds: float) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.5)
    return read_from(log_path, offset)


def has_clear_during_playback(text: str) -> bool:
    for play in PLAY_RE.finditer(text):
        done = PLAY_DONE_RE.search(text, play.end())
        done_pos = done.start() if done else len(text)
        for clear in BASE_TTS_CLEAR_RE.finditer(text, play.end(), done_pos):
            return True
        for clear in PLAYER_CLEAR_RE.finditer(text, play.end(), done_pos):
            return True
    return False


def evaluate(idx: int, message: str, text: str) -> WindowResult:
    attention_ids = ATTENTION_RE.findall(text)
    settled_count = len(SETTLED_RE.findall(text))
    tts_stream_count = len(TTS_STREAM_RE.findall(text))
    play_positions = [m.start() for m in PLAY_RE.finditer(text)]
    body_commands = BODY_COMMAND_RE.findall(text)
    sync_matches = list(SPEECH_SYNC_RE.finditer(text))
    sync_commands = [m.group(1) for m in sync_matches]
    timeout_commands = SPEECH_TIMEOUT_RE.findall(text)

    failures: list[str] = []
    if len(attention_ids) != 1:
        failures.append(f"expected 1 set attention, got {len(attention_ids)} ids={attention_ids}")
    elif attention_ids.count(attention_ids[0]) > 1:
        failures.append(f"duplicate attention id {attention_ids[0]}")

    if settled_count != 1:
        failures.append(f"expected 1 interpreter settled, got {settled_count}")

    if tts_stream_count != 1:
        failures.append(f"expected 1 TTS stream, got {tts_stream_count}")

    if not play_positions:
        failures.append("expected at least 1 uploaded audio playback")

    if has_clear_during_playback(text):
        failures.append("TTS/player clear occurred during upload playback")

    if timeout_commands:
        failures.append(f"body action timed out waiting for speech: {timeout_commands}")

    sync_required = [command for command in body_commands if command in SYNC_REQUIRED_COMMANDS]
    if sync_required:
        if not play_positions:
            failures.append(f"body commands without audio playback: {sync_required}")
        if not sync_matches:
            failures.append(f"body commands without speech sync logs: {sync_required}")
        else:
            first_play = play_positions[0] if play_positions else len(text)
            early_sync = [m.group(1) for m in sync_matches if m.start() < first_play]
            if early_sync:
                failures.append(f"body sync happened before first playback log: {early_sync}")

    return WindowResult(
        idx=idx,
        message=message,
        attention_ids=attention_ids,
        settled_count=settled_count,
        tts_stream_count=tts_stream_count,
        play_count=len(play_positions),
        body_commands=body_commands,
        sync_commands=sync_commands,
        failures=failures,
    )


def build_messages(args: argparse.Namespace) -> list[str]:
    if args.message:
        return args.message
    return DEFAULT_MESSAGES[: args.count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="default")
    parser.add_argument("--log", default=".moss_ws/runtime/logs/moss.log")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--settle-timeout", type=float, default=90.0)
    parser.add_argument("--repeat-window", type=float, default=30.0)
    parser.add_argument(
        "--bus-warmup",
        type=float,
        default=2.0,
        help="seconds to keep the fresh Zenoh session open before the first publish",
    )
    parser.add_argument("--message", action="append", help="message to publish; repeat for multiple")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"FAIL: log file not found: {log_path}", file=sys.stderr)
        return 2

    run_id = uuid4().hex[:8]
    messages = build_messages(args)
    failures: list[WindowResult] = []

    print(f"smoke run_id={run_id} scope={args.scope} messages={len(messages)}", flush=True)
    with zenoh.open(zenoh.Config()) as z:
        if args.bus_warmup > 0:
            time.sleep(args.bus_warmup)
        for idx, message in enumerate(messages, 1):
            offset = log_path.stat().st_size
            description = f"moss-smoke:{run_id}:{idx}"
            print(f"\n[{idx}/{len(messages)}] publish {description}: {message}", flush=True)
            publish_input(z, args.scope, message, description)

            text = wait_for_settled(log_path, offset, args.settle_timeout)
            if not SETTLED_RE.search(text):
                result = evaluate(idx, message, text)
                result.failures.append(f"timed out waiting for interpreter settled after {args.settle_timeout:.1f}s")
                failures.append(result)
                print("  FAIL: timed out waiting for settled", flush=True)
                continue

            text = collect_repeat_window(log_path, offset, args.repeat_window)
            result = evaluate(idx, message, text)
            status = "PASS" if not result.failures else "FAIL"
            print(
                "  "
                f"{status}: attention={result.attention_ids} settled={result.settled_count} "
                f"tts_streams={result.tts_stream_count} uploads={result.play_count} "
                f"body={result.body_commands} sync={result.sync_commands}",
                flush=True,
            )
            for failure in result.failures:
                print(f"    - {failure}", flush=True)
            if result.failures:
                failures.append(result)

    if failures:
        print(f"\nFAIL: {len(failures)} window(s) failed", flush=True)
        return 1

    print("\nPASS: text loop smoke completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
