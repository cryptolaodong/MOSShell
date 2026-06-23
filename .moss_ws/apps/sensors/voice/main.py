import asyncio
# 禁用 GStreamer 避免 ReachyMini SDK 加载时 segfault
import os
os.environ['GST_PLUGIN_PATH'] = ''
os.environ['GST_PLUGIN_SYSTEM_PATH'] = ''

import json
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
# pynput removed - using VAD auto-listen mode
from rich.console import Console
from rich.panel import Panel

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.asr.async_concepts import (
    AsyncListenerService,
    AsyncListenerCallback,
    AsyncListenerStateName,
    Recognition,
)
from ghoshell_moss_contrib.asr.async_listener_service import AsyncListenerServiceImpl
from ghoshell_moss_contrib.asr.configs import ListenerConfig
from ghoshell_moss_contrib.asr.voice_turn_gate import VoiceTurnDecision, VoiceTurnGate
from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speaking_gate import (
    is_speaking as robot_is_speaking,
    mark_thinking as robot_mark_thinking,
    remaining_seconds as robot_speaking_remaining_seconds,
)

load_dotenv()

# VAD auto-listen mode - no PTT key needed


def _truthy_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _apply_reachy_mic_asr_defaults(mic_backend_selected: str) -> None:
    if mic_backend_selected not in {"reachy_robot", "auto_reachy_then_local"}:
        return
    force_defaults = _truthy_env("MOSS_REACHY_MIC_FORCE_ASR_DEFAULTS", True)
    defaults = {
        # The robot mic hears mechanical/environment bursts. Local tiny Whisper
        # can hallucinate wake words on those bursts, and long empty cooldowns
        # make real short turns feel like the robot ignored the user.
        "MOSS_ASR_LOCAL_FALLBACK_ENABLED": "1",
        "MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE": "1",
        "MOSS_ASR_LOCAL_FALLBACK_INITIAL_PROMPT": "",
        "MOSS_ASR_LOCAL_FALLBACK_RESCUE_PROMPT": "小白你好",
        "MOSS_ASR_LOCAL_FALLBACK_RESCUE_MODEL": "base",
        "MOSS_VOICE_ADDRESS_WORDS": "小白,蒋白",
        "MOSS_ASR_LOCAL_FALLBACK_MIN_RMS": "1800",
        "MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS": "0.9",
        "MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS": "0.85",
        "MOSS_ASR_LOCAL_FALLBACK_TIMEOUT_SECONDS": "3.0",
        "MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS": "4.2",
        "MOSS_ASR_LONG_STABLE_TEXT_COMMIT_SECONDS": "1.8",
        "MOSS_ASR_STABLE_TEXT_MIN_QUIET_SECONDS": "1.2",
        "MOSS_ASR_LONG_EMPTY_TEXT_COMMIT_SECONDS": "1.8",
        "MOSS_ASR_LONG_EMPTY_TEXT_MIN_QUIET_SECONDS": "1.2",
        "MOSS_ASR_NO_TEXT_COOLDOWN_SECONDS": "0",
        "MOSS_ASR_NO_TEXT_COOLDOWN_FACTOR": "1.0",
        "MOSS_ASR_NO_TEXT_COOLDOWN_MAX_SECONDS": "0",
        "MOSS_ASR_ENERGY_SPEECH_RMS": "1800",
        "MOSS_ASR_INPUT_GATE_RMS": "1800",
        "MOSS_ASR_INPUT_GATE_OPEN_FRAMES": "1",
        "MOSS_ASR_SPEECH_NO_TEXT_MAX_SECONDS": "3.2",
        "MOSS_ASR_SPEECH_NO_TEXT_HARD_MULTIPLIER": "1.6",
        "MOSS_ASR_PRESPEECH_BATCH_MAX_SECONDS": "3.2",
        "MOSS_ASR_INPUT_GATE_PREROLL_SECONDS": "2.0",
        "MOSS_ASR_INPUT_GATE_TAIL_SECONDS": "1.2",
        "MOSS_VOICE_SAVE_EMPTY_ASR_AUDIO": "0",
        "MOSS_VOICE_SAVE_REJECTED_ASR_AUDIO": "1",
        "MOSS_VOICE_REJECTED_ASR_AUDIO_MIN_RMS": "1800",
    }
    applied: dict[str, str] = {}
    for name, value in defaults.items():
        if force_defaults or name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    _latency_log(
        "voice_reachy_mic_asr_defaults",
        selected=mic_backend_selected,
        forced=force_defaults,
        applied=applied,
    )


def _log_reachy_daemon_status(robot_host: str) -> None:
    try:
        from urllib import request

        with request.urlopen(f"http://{robot_host}:8000/api/daemon/status", timeout=2.0) as resp:
            status = json.loads(resp.read().decode("utf-8"))
    except Exception as error:
        _latency_log(
            "voice_reachy_daemon_status_error",
            host=robot_host,
            error=str(error)[:200],
        )
        return
    _latency_log(
        "voice_reachy_daemon_status",
        host=robot_host,
        state=status.get("state"),
        error=status.get("error"),
        media_released=status.get("media_released"),
        no_media=status.get("no_media"),
        wlan_ip=status.get("wlan_ip"),
        version=status.get("version"),
    )


def _workspace_root_for_logs() -> Path | None:
    workspace = os.environ.get("MOSS_WORKSPACE", "").strip()
    if workspace:
        return Path(workspace)
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == ".moss_ws":
            return parent
    return None


def _latency_log_path() -> Path:
    configured = os.environ.get("MOSS_VOICE_LATENCY_LOG", "").strip()
    workspace = _workspace_root_for_logs()
    if configured:
        path = Path(configured)
    elif workspace:
        path = workspace / "runtime" / "logs" / "voice_latency.log"
    else:
        path = Path(".moss_ws/runtime/logs/voice_latency.log")
    if not path.is_absolute() and workspace:
        if path.parts and path.parts[0] == ".moss_ws":
            path = workspace.parent.joinpath(*path.parts)
        else:
            path = workspace / path
    return path


def _latency_log(event: str, **fields) -> None:
    path = _latency_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": event, "ts": time.time(), **fields}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def _voice_debug_audio_dir() -> Path:
    configured = os.environ.get("MOSS_VOICE_DEBUG_AUDIO_DIR", "").strip()
    workspace = _workspace_root_for_logs()
    if configured:
        path = Path(configured)
    elif workspace:
        path = workspace / "runtime" / "voice_debug"
    else:
        path = Path(".moss_ws/runtime/voice_debug")
    if not path.is_absolute() and workspace:
        path = workspace / path
    return path


def _audio_rms_peak(audio: np.ndarray) -> tuple[float, int]:
    if audio is None or len(audio) <= 0:
        return 0.0, 0
    flat = np.asarray(audio).reshape(-1)
    if len(flat) <= 0:
        return 0.0, 0
    values = flat.astype(float)
    rms = float(np.sqrt(np.mean(values ** 2)))
    peak = int(np.max(np.abs(values)))
    return rms, peak


def _write_debug_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    flat = np.asarray(audio).reshape(-1)
    if flat.dtype != np.int16:
        flat = np.clip(flat, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(flat.tobytes())

# ── ThreadedListenerService (extracted from ConsolePTTChat) ──────────────


class ThreadedListenerService:
    """在独立线程中运行 Listener 服务，与主 event loop 完全解耦。"""

    def __init__(self, inner: AsyncListenerService, main_loop: asyncio.AbstractEventLoop, logger):
        self._inner = inner
        self._main_loop = main_loop
        self._logger = logger
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _schedule(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _await_schedule(self, coro):
        future = self._schedule(coro)
        return await asyncio.wrap_future(future)

    async def _forward_callback(self, callback):
        class _ForwardCallback(AsyncListenerCallback):
            def __init__(self, cb, main_loop, logger):
                self._cb = cb
                self._main_loop = main_loop
                self._logger = logger

            async def on_recognition(self, result):
                asyncio.run_coroutine_threadsafe(self._cb.on_recognition(result), self._main_loop)

            async def on_state_change(self, state):
                asyncio.run_coroutine_threadsafe(self._cb.on_state_change(state), self._main_loop)

            async def on_waken(self):
                asyncio.run_coroutine_threadsafe(self._cb.on_waken(), self._main_loop)

            async def on_error(self, error):
                asyncio.run_coroutine_threadsafe(self._cb.on_error(error), self._main_loop)

            async def save_batch(self, rec, audio):
                asyncio.run_coroutine_threadsafe(self._cb.save_batch(rec, audio), self._main_loop)

        return _ForwardCallback(callback, self._main_loop, self._logger)

    async def bootstrap(self):
        ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._inner.bootstrap())
                ready.set()
                self._loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="voice-listener")
        self._thread.start()
        ready.wait(timeout=10)
        if not ready.is_set():
            raise RuntimeError("Listener service failed to bootstrap within 10s")
        self._logger.info("ThreadedListenerService started")

    async def shutdown(self):
        if self._loop and self._loop.is_running():
            try:
                await self._await_schedule(self._inner.shutdown())
            except Exception as e:
                self._logger.warning(f"Listener shutdown error: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._logger.info("ThreadedListenerService shutdown complete")

    async def set_callback(self, callback):
        forwarded = await self._forward_callback(callback)
        await self._await_schedule(self._inner.set_callback(forwarded))

    async def set_state(self, state: str):
        self._schedule(self._inner.set_state(state))

    async def current_state(self):
        return await self._await_schedule(self._inner.current_state())

    async def commit(self):
        self._schedule(self._inner.commit())

    async def clear_buffer(self):
        self._schedule(self._inner.clear_buffer())


# ── Voice Status Display ──────────────────────────────────────────────────


class VoiceStatusDisplay:
    """最小状态 TUI：区分识别中（流式刷新）和识别完成。"""

    def __init__(self):
        self.console = Console()
        self.state = "idle"
        self._partial_printed = False  # 跟踪是否有未闭合的 partial 行

    def show_header(self):
        self.console.print(Panel("[bold cyan]Voice Input App[/bold cyan] — input/voice", title="MOSS"))

    def show_state(self, state: str):
        """状态切换，相同状态不重复打印"""
        if state == self.state:
            return
        # 如果之前在流式打印 partial，先换行收尾
        if self._partial_printed:
            self.console.print()
            self._partial_printed = False
        icons = {"idle": "⏸", "recording": "🔴", "sending": "📤"}
        labels = {"idle": "Waiting", "recording": "Recording", "sending": "Sending"}
        self.state = state
        self.console.print(f"  {icons.get(state, '?')} [bold]{labels.get(state, state)}[/bold]")

    def show_partial(self, text: str):
        """流式覆盖打印当前识别中的部分文本"""
        # console.print + end="\r" 让 Rich 正确渲染颜色，同时不换行
        self.console.print(f"  [green]→ {text}[/green]", end="\r")
        self._partial_printed = True

    def show_recognized(self, text: str, reason: str = ""):
        """识别完成，清除 partial 行后打印最终结果"""
        reason_tag = f" [{reason}]" if reason else ""
        if self._partial_printed:
            self.console.file.write("\r\033[K")
            self.console.file.flush()
        self.console.print(f"  [cyan]✓ {text}{reason_tag}[/cyan]")
        self._partial_printed = False

    def show_sent(self):
        self.console.print(f"  [dim]📤 sent to ghost[/dim]")

    def show_footer(self):
        self.console.print("[dim]Auto-listen mode. Speak freely! Ctrl+C to quit[/dim]")


# ── Main Entry ────────────────────────────────────────────────────────────


async def _toggle_recording(threaded: ThreadedListenerService):
    try:
        current_state = await threaded.current_state()
        state_name = current_state.name().value
    except Exception:
        return

    if state_name == AsyncListenerStateName.PDT_LISTENING.value:
        await threaded.commit()
    else:
        await threaded.set_state(AsyncListenerStateName.PDT_LISTENING.value)


async def main(matrix: Matrix) -> None:
    console = Console()
    logger = logging.getLogger("VoiceInput")
    logging.basicConfig(level=logging.WARNING)

    display = VoiceStatusDisplay()
    display.show_header()
    console.print("[green]Initializing voice input pipeline...[/green]")

    # 1. Listener config + service
    config = ListenerConfig()
    from sd_audio_input import ReachyMicAudioInput, SoundDeviceAudioInput

    mic_backend = os.environ.get("MOSS_VOICE_INPUT_BACKEND", "auto").strip().lower()
    if mic_backend in {"local", "sounddevice"}:
        console.print("[cyan]Voice mic backend: local sounddevice[/cyan]")
        sd_input = SoundDeviceAudioInput(rate=16000, channels=1)
        mic_backend_selected = "local_sounddevice"
    elif mic_backend in {"reachy", "robot"}:
        console.print("[cyan]Voice mic backend: Reachy robot microphone[/cyan]")
        os.environ.setdefault("MOSS_REACHY_MIC_FALLBACK", "0")
        sd_input = ReachyMicAudioInput(rate=16000, channels=1)
        mic_backend_selected = "reachy_robot"
    else:
        console.print("[cyan]Voice mic backend: auto (Reachy robot mic, then local fallback)[/cyan]")
        sd_input = ReachyMicAudioInput(rate=16000, channels=1)
        mic_backend_selected = "auto_reachy_then_local"
    if mic_backend_selected in {"reachy_robot", "auto_reachy_then_local"}:
        _log_reachy_daemon_status(os.environ.get("REACHY_ROBOT_HOST", "reachy-mini.local"))
    _apply_reachy_mic_asr_defaults(mic_backend_selected)
    _latency_log(
        "voice_mic_backend_selected",
        requested=mic_backend or "auto",
        selected=mic_backend_selected,
        input_id=getattr(sd_input, "input_id", ""),
    )

    inner = AsyncListenerServiceImpl(
        config=config,
        logger=logger,
        audio_input=sd_input,
    )

    main_loop = asyncio.get_running_loop()
    threaded = ThreadedListenerService(inner, main_loop, logger)
    await threaded.bootstrap()
    _latency_log(
        "voice_listener_bootstrap_done",
        selected=mic_backend_selected,
        input_id=getattr(sd_input, "input_id", ""),
    )

    # 2. Callback: ASR text → session.add_input_signal()
    import time as _time

    _dedup = {"text": "", "ts": 0.0}  # 防重复 send
    require_address = _truthy_env("MOSS_VOICE_REQUIRE_ADDRESS_WHEN_IDLE", True)
    active_seconds = float(os.environ.get("MOSS_VOICE_SESSION_ACTIVE_SECONDS", "12"))
    robot_speaking_tail_seconds = float(os.environ.get("MOSS_VOICE_ROBOT_SPEAKING_TAIL_SECONDS", "3.5"))
    address_words = tuple(
        word.strip()
        for word in os.environ.get("MOSS_VOICE_ADDRESS_WORDS", "小白").split(",")
        if word.strip()
    )
    idle_partial_chars = int(os.environ.get("MOSS_VOICE_IDLE_UNADDRESSED_PARTIAL_CHARS", "999"))
    idle_partial_seconds = float(os.environ.get("MOSS_VOICE_IDLE_UNADDRESSED_PARTIAL_SECONDS", "1.2"))
    local_fallback_address_min_rms = float(
        os.environ.get("MOSS_VOICE_LOCAL_FALLBACK_ADDRESS_MIN_RMS", "1200")
    )
    local_fallback_followup_min_rms = float(
        os.environ.get("MOSS_VOICE_LOCAL_FALLBACK_FOLLOWUP_MIN_RMS", "900")
    )
    wake_recovery_seconds = float(os.environ.get("MOSS_VOICE_WAKE_RECOVERY_SECONDS", "3.0"))
    wake_recovery_min_rms = float(os.environ.get("MOSS_VOICE_WAKE_RECOVERY_MIN_RMS", "1800"))
    wake_recovery_tokens = tuple(
        token.strip()
        for token in os.environ.get(
            "MOSS_VOICE_WAKE_RECOVERY_TOKENS",
            "你好,在吗,在不在,能做什么,会做什么,能做,做什么,动动,摇头,点头,转头,抬头,低头,跳舞,表情,介绍一下,你是谁",
        ).split(",")
        if token.strip()
    )
    turn_gate = VoiceTurnGate(
        require_address_when_idle=require_address,
        address_words=address_words,
        active_seconds=active_seconds,
        idle_partial_chars=idle_partial_chars,
        idle_partial_seconds=idle_partial_seconds,
    )
    _wake_recovery = {"until": 0.0, "reason": "", "rms": 0.0}
    control_file = Path(os.environ.get("MOSS_VOICE_CONTROL_FILE", "/private/tmp/moss_voice_control.json"))
    _control = {"mtime": 0.0}
    save_empty_audio = _truthy_env("MOSS_VOICE_SAVE_EMPTY_ASR_AUDIO", True)
    save_all_audio = _truthy_env("MOSS_VOICE_SAVE_ALL_ASR_AUDIO", False)
    save_rejected_audio = _truthy_env("MOSS_VOICE_SAVE_REJECTED_ASR_AUDIO", False)
    debug_audio_min_rms = float(os.environ.get("MOSS_VOICE_DEBUG_AUDIO_MIN_RMS", "500"))
    rejected_audio_min_rms = float(os.environ.get("MOSS_VOICE_REJECTED_ASR_AUDIO_MIN_RMS", "1800"))
    debug_audio_max_seconds = float(os.environ.get("MOSS_VOICE_DEBUG_AUDIO_MAX_SECONDS", "12"))
    debug_audio_dir = _voice_debug_audio_dir()

    async def _poll_control_file() -> None:
        try:
            stat = control_file.stat()
        except FileNotFoundError:
            return
        except Exception:
            return
        if stat.st_mtime <= _control["mtime"]:
            return
        _control["mtime"] = stat.st_mtime
        try:
            payload = json.loads(control_file.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        action = str(payload.get("action") or "").strip().lower()
        if action == "clear":
            await threaded.clear_buffer()
            _wake_recovery["until"] = 0.0
            display.show_state("idle")
            _latency_log(
                "voice_control_clear",
                nonce=str(payload.get("nonce") or "")[:80],
                source=str(payload.get("source") or "")[:80],
            )

    def _is_wake_recovery_followup_text(text: str) -> bool:
        normalized = (text or "").strip().lower()
        normalized = normalized.strip(" \t\r\n，,。！？!?；;：:")
        normalized = normalized.replace(" ", "")
        if not normalized or len(normalized) > 24:
            return False
        return any(token in normalized for token in wake_recovery_tokens)

    class VoiceCallback(AsyncListenerCallback):
        async def on_recognition(self, result: Recognition):
            if not result.text or not result.text.strip():
                return
            if result.is_last:
                text = result.text.strip()
                if robot_is_speaking(tail=robot_speaking_tail_seconds):
                    logger.info(
                        "[VoiceInput] drop ASR while robot speaking: %.1fs left, text=%r",
                        robot_speaking_remaining_seconds(),
                        text[:80],
                    )
                    _latency_log(
                        "voice_drop_robot_speaking",
                        left_seconds=round(robot_speaking_remaining_seconds(), 3),
                        text_len=len(text),
                        reason=result.commit_reason or "",
                        text_preview=text[:40],
                    )
                    await threaded.clear_buffer()
                    display.show_state("idle")
                    return
                now = _time.monotonic()
                is_local_fallback = (result.commit_reason or "").startswith("local_whisper")
                audio_max_rms = float(getattr(result, "audio_max_rms", 0.0) or 0.0)
                addressed_before = turn_gate.is_addressed(text)
                active_left_before = turn_gate.active_left(now)
                active_before = active_left_before > 0
                recovery_left = max(0.0, float(_wake_recovery.get("until", 0.0)) - now)
                recovery_allowed = (
                    is_local_fallback
                    and not addressed_before
                    and not active_before
                    and recovery_left > 0
                    and audio_max_rms >= wake_recovery_min_rms
                    and _is_wake_recovery_followup_text(text)
                )
                weak_local_fallback = False
                if is_local_fallback and (addressed_before or active_before):
                    min_rms = (
                        local_fallback_address_min_rms
                        if addressed_before
                        else local_fallback_followup_min_rms
                    )
                    weak_local_fallback = audio_max_rms > 0 and audio_max_rms < min_rms
                if weak_local_fallback:
                    turn_gate.reset_idle_partial()
                    gate_decision = VoiceTurnDecision(
                        accept=False,
                        reason="local_fallback_weak_audio",
                        addressed=addressed_before,
                        active_before=active_before,
                        active_left_seconds=round(active_left_before, 3),
                        text_len=len(text),
                    )
                elif recovery_allowed:
                    turn_gate.open_followup_window(now)
                    _wake_recovery["until"] = 0.0
                    gate_decision = VoiceTurnDecision(
                        accept=True,
                        reason="wake_recovery_followup",
                        addressed=False,
                        active_before=False,
                        active_left_seconds=round(recovery_left, 3),
                        text_len=len(text),
                    )
                    _latency_log(
                        "voice_wake_recovery_accept",
                        text_len=len(text),
                        text_preview=text[:40],
                        audio_max_rms=round(audio_max_rms, 1),
                        recovery_left=round(recovery_left, 3),
                        armed_reason=str(_wake_recovery.get("reason") or "")[:60],
                        armed_rms=round(float(_wake_recovery.get("rms", 0.0) or 0.0), 1),
                    )
                else:
                    gate_decision = turn_gate.decide_final(text, now)
                if not gate_decision.accept:
                    logger.info("[VoiceInput] drop idle background ASR: text=%r", text[:80])
                    _latency_log(
                        "voice_drop_not_addressed",
                        text_len=len(text),
                        reason=result.commit_reason or "",
                        text_preview=text[:40],
                        audio_max_rms=round(audio_max_rms, 1),
                        gate_reason=gate_decision.reason,
                        addressed=gate_decision.addressed,
                        active_before=gate_decision.active_before,
                        active_left_seconds=gate_decision.active_left_seconds,
                    )
                    _latency_log(
                        "voice_gate_drop",
                        text_len=len(text),
                        asr_reason=result.commit_reason or "",
                        text_preview=text[:40],
                        audio_max_rms=round(audio_max_rms, 1),
                        gate_reason=gate_decision.reason,
                        addressed=gate_decision.addressed,
                        active_before=gate_decision.active_before,
                        active_left_seconds=gate_decision.active_left_seconds,
                    )
                    await threaded.clear_buffer()
                    display.show_state("idle")
                    return

                display.show_recognized(text, result.commit_reason or "")
                display.show_state("sending")
                # 防止 VAD auto-commit 和手动 commit 重复发送同一句
                if not (text == _dedup["text"] and now - _dedup["ts"] < 1.5):
                    sent_started = _time.monotonic()
                    _latency_log(
                        "voice_final_before_send",
                        text_len=len(text),
                        reason=result.commit_reason or "",
                        text_preview=text[:40],
                        audio_max_rms=round(audio_max_rms, 1),
                        gate_reason=gate_decision.reason,
                        addressed=gate_decision.addressed,
                        active_before=gate_decision.active_before,
                        active_left_seconds=gate_decision.active_left_seconds,
                    )
                    robot_mark_thinking()
                    matrix.session.add_input_signal(
                        text,
                        description=f"voice: {text[:50]}",
                    )
                    logger.warning(
                        "[ReachyLatency] voice_final_sent text_len=%d reason=%s submit_elapsed=%.3fs",
                        len(text),
                        result.commit_reason or "",
                        _time.monotonic() - sent_started,
                    )
                    _latency_log(
                        "voice_final_sent",
                        text_len=len(text),
                        reason=result.commit_reason or "",
                        submit_elapsed=round(_time.monotonic() - sent_started, 3),
                        text_preview=text[:40],
                        audio_max_rms=round(audio_max_rms, 1),
                        gate_reason=gate_decision.reason,
                        addressed=gate_decision.addressed,
                        active_before=gate_decision.active_before,
                        active_left_seconds=gate_decision.active_left_seconds,
                    )
                    _dedup["text"] = text
                    _dedup["ts"] = now
                else:
                    _latency_log(
                        "voice_duplicate_dropped",
                        text_len=len(text),
                        reason=result.commit_reason or "",
                        since_last=round(now - _dedup["ts"], 3),
                        text_preview=text[:40],
                    )
                display.show_sent()
                display.show_state("idle")
                display.show_footer()
            else:
                text = result.text.strip()
                now = _time.monotonic()
                partial_decision = turn_gate.decide_partial(text, now)
                if partial_decision.commit:
                    logger.info(
                        "[VoiceInput] commit idle unaddressed partial early: age=%.2fs text=%r",
                        partial_decision.age_seconds,
                        text[:80],
                    )
                    _latency_log(
                        "voice_idle_partial_unaddressed_commit",
                        age=partial_decision.age_seconds,
                        text_len=len(text),
                        text_preview=text[:40],
                        gate_reason=partial_decision.reason,
                        addressed=partial_decision.addressed,
                        active_before=partial_decision.active_before,
                        active_left_seconds=partial_decision.active_left_seconds,
                    )
                    await threaded.commit()
                    return
                display.show_partial(text)

        async def on_state_change(self, state: str):
            if "listening" in state.lower():
                display.show_state("recording")
            # 不处理 waiting → idle，让 on_recognition(is_last=True) 统一收尾

        async def on_error(self, error: str):
            _latency_log("voice_error", error=str(error)[:200])
            display.console.print(f"  [red]❌ {error}[/red]")

        async def on_waken(self):
            pass

        async def save_batch(self, rec: Recognition, audio: np.ndarray):
            if audio is None or len(audio) <= 0:
                return
            flat = np.asarray(audio).reshape(-1)
            if len(flat) <= 0:
                return
            rms, peak = _audio_rms_peak(flat)
            text = (rec.text or "").strip()
            reason = rec.commit_reason or ""
            rejected_reason = reason in {
                "local_fallback_no_safe_text",
                "speech_no_text_timeout",
            } or (not text and reason in {
                "energy_vad",
                "audio_idle",
                "empty_text_timeout",
                "empty_speech_commit",
            })
            try:
                rec_max_rms = float(getattr(rec, "audio_max_rms", 0.0) or 0.0)
            except (TypeError, ValueError):
                rec_max_rms = 0.0
            diagnostic_rms = max(rms, rec_max_rms)
            should_save = (
                save_all_audio
                or (save_empty_audio and not text and rms >= debug_audio_min_rms)
                or (save_rejected_audio and rejected_reason and diagnostic_rms >= rejected_audio_min_rms)
            )
            if not should_save:
                return
            sample_rate = 16000
            if debug_audio_max_seconds > 0:
                max_samples = int(sample_rate * debug_audio_max_seconds)
                if len(flat) > max_samples:
                    flat = flat[-max_samples:]
            safe_reason = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in reason)[:40]
            filename = (
                f"{time.strftime('%Y%m%d-%H%M%S')}-"
                f"{rec.batch_id[:8]}-{safe_reason or 'asr'}.wav"
            )
            wav_path = debug_audio_dir / filename
            try:
                _write_debug_wav(wav_path, flat, sample_rate=sample_rate)
                meta_path = wav_path.with_suffix(".json")
                meta = {
                    "batch_id": rec.batch_id,
                    "commit_reason": reason,
                    "text_len": len(text),
                    "is_last": rec.is_last,
                    "rms": round(rms, 2),
                    "diagnostic_rms": round(diagnostic_rms, 2),
                    "peak": peak,
                    "sample_rate": sample_rate,
                    "samples": int(len(flat)),
                    "duration_seconds": round(len(flat) / sample_rate, 3),
                    "save_rejected_audio": bool(save_rejected_audio and rejected_reason),
                    "created_at": time.time(),
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                _latency_log(
                    "voice_debug_audio_saved",
                    path=str(wav_path),
                    reason=reason,
                    text_len=len(text),
                    rms=round(rms, 2),
                    diagnostic_rms=round(diagnostic_rms, 2),
                    peak=peak,
                    duration=round(len(flat) / sample_rate, 3),
                )
                if (
                    wake_recovery_seconds > 0
                    and not text
                    and rejected_reason
                    and diagnostic_rms >= wake_recovery_min_rms
                ):
                    _wake_recovery["until"] = _time.monotonic() + wake_recovery_seconds
                    _wake_recovery["reason"] = reason
                    _wake_recovery["rms"] = diagnostic_rms
                    _latency_log(
                        "voice_wake_recovery_armed",
                        reason=reason,
                        diagnostic_rms=round(diagnostic_rms, 1),
                        seconds=round(wake_recovery_seconds, 3),
                    )
            except Exception as e:
                _latency_log("voice_debug_audio_save_error", error=str(e)[:200])

    await threaded.set_callback(VoiceCallback())

    # 3. Auto-listen mode (VAD) — 持续监听，不需要按键
    console.print("[green]Auto-listen mode started (VAD). Speak freely![/green]")
    display.show_state("listening")
    display.show_footer()

    # 自动进入监听状态
    await threaded.set_state(AsyncListenerStateName.PDT_LISTENING.value)

    # 4. 持续循环：监听结束后自动重新开始监听
    poll_interval = float(os.environ.get("MOSS_VOICE_LOOP_POLL_SECONDS", "0.15"))
    restart_delay = float(os.environ.get("MOSS_VOICE_RESTART_DELAY_SECONDS", "0.08"))
    speaking_tail = float(os.environ.get("MOSS_VOICE_ROBOT_SPEAKING_TAIL_SECONDS", "0.8"))
    try:
        while True:
            await asyncio.sleep(poll_interval)
            try:
                await _poll_control_file()
                if robot_is_speaking(tail=speaking_tail):
                    await threaded.clear_buffer()
                    continue
                current_state = await threaded.current_state()
                state_name = current_state.name().value
                # 如果不在监听状态，自动重新开始监听
                if state_name != AsyncListenerStateName.PDT_LISTENING.value:
                    await asyncio.sleep(restart_delay)  # 短暂间隔避免打断
                    await threaded.set_state(AsyncListenerStateName.PDT_LISTENING.value)
            except Exception:
                await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        pass
    finally:
        console.print("[yellow]Shutting down...[/yellow]")
        await threaded.shutdown()
        console.print("[green]Voice input app stopped.[/green]")


if __name__ == "__main__":
    try:
        Matrix.discover().run(main)
    except KeyboardInterrupt:
        print("\nVoice input app stopped.")
