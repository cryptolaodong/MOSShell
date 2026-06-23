"""
异步 Listener 状态实现。

完全基于 asyncio，弃用线程模型。
"""

import asyncio
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Union, Callable
import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid, Timeleft

from .async_concepts import (
    AsyncListenerState,
    AsyncListenerStateName,
    AsyncListenerCallback,
    AsyncRecognizer,
    AsyncRecognitionBatch,
    AsyncAudioInput,
    Recognition,
    AsyncRecognitionCallback,
)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _ends_terminal_punctuation(text: str) -> bool:
    return text.rstrip().endswith(("。", "？", "?", "！", "!", "；", ";", ".", "…"))


_LOCAL_WHISPER_LOCK = threading.Lock()
_LOCAL_WHISPER_MODEL = None
_LOCAL_WHISPER_MODEL_KEY: tuple[str, str, str] | None = None
_NO_TEXT_COOLDOWN_UNTIL = 0.0
_NO_TEXT_FAILURE_COUNT = 0


def _clean_local_asr_text(text: str) -> str:
    return (text or "").strip().strip(" \t\r\n，,。！？!?；;：:")


def _normalize_local_asr_text(text: str) -> str:
    normalized = _clean_local_asr_text(text).lower()
    replacements = {
        "現": "现",
        "麼": "么",
        "什麼": "什么",
        "嗎": "吗",
        "會": "会",
        "臺": "台",
        "妳": "你",
        "線": "线",
        "夠": "够",
        "號": "号",
        "魚": "鱼",
        "癢": "痒",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    for mark in ("，", ",", "。", "！", "!", "？", "?", "；", ";", "：", ":"):
        normalized = normalized.replace(mark, "")
    return normalized.replace(" ", "")


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[index_b] + 1,
                    current[index_b - 1] + 1,
                    previous[index_b - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _looks_like_short_wake_greeting(normalized: str) -> bool:
    if not 3 <= len(normalized) <= 5:
        return False
    if "白" not in normalized or "好" not in normalized:
        return False
    return _edit_distance(normalized, "小白你好") <= 2


def _looks_like_rescuable_short_wake_fragment(normalized: str) -> bool:
    if not 2 <= len(normalized) <= 5:
        return False
    if any(blocked in normalized for blocked in ("酒", "明", "铃", "鈴")):
        return False
    if normalized in {"小一号", "小孩好"}:
        return True
    if "白" not in normalized:
        return False
    return any(token in normalized for token in ("你", "好", "号", "痒", "吧"))


def _looks_like_rescuable_wake_second_pass(normalized: str) -> bool:
    if not 2 <= len(normalized) <= 8:
        return False
    if any(blocked in normalized for blocked in ("酒", "明", "铃", "鈴")):
        return False
    if normalized in {"小雷"}:
        return True
    if not normalized.startswith(("小", "想", "叫", "老")):
        return False
    return any(token in normalized for token in ("你好", "好", "号", "號"))


def _canonicalize_safe_local_fallback_text(text: str) -> str:
    cleaned = _clean_local_asr_text(text)
    normalized = _normalize_local_asr_text(cleaned)
    exact_homophones = {
        "小番茗好": "小白你好",
        "小白米好": "小白你好",
        "小白糖": "小白你好",
        "小白一号": "小白你好",
        "老白你好": "小白你好",
    }
    if normalized in exact_homophones:
        return exact_homophones[normalized]
    if _looks_like_short_wake_greeting(normalized):
        return "小白你好"
    wake_homophones = ("想掰", "想把", "想法", "小拜", "小摆", "小百", "小班")
    for wake in wake_homophones:
        if normalized.startswith(wake) and len(normalized) <= 16:
            rest = normalized[len(wake):]
            if any(token in rest for token in ("你好", "在吗", "在不在", "哈喽", "hello")):
                return f"小白{rest}"
            if any(
                token in rest
                for token in ("能做", "会做", "做什么", "能够什么", "能干什么", "干什么")
            ):
                return "小白你现在能做什么"
    if normalized.startswith("小白") and len(normalized) <= 16:
        rest = normalized[2:]
        if any(
            token in rest
            for token in ("能做", "会做", "做什么", "能够什么", "能干什么", "干什么")
        ):
            return "小白你现在能做什么"
    if normalized == "你现在能做什么":
        return "你现在能做什么"
    return cleaned


def _is_safe_local_fallback_text(text: str) -> bool:
    normalized = _normalize_local_asr_text(_canonicalize_safe_local_fallback_text(text))
    if len(normalized) <= 1:
        return False
    exact_phrases = {
        "小白",
        "小白你好",
        "你好小白",
        "小白在吗",
        "小白在不在",
        "在吗小白",
        "你现在能做什么",
        "你能做什么",
        "现在能做什么",
        "能做什么",
        "你会做什么",
        "你是谁",
    }
    if normalized in exact_phrases:
        return True
    if normalized.startswith("小白") and len(normalized) <= 16:
        return any(
            token in normalized
            for token in (
                "你好",
                "在吗",
                "在不在",
                "哈喽",
                "hello",
                "能做什么",
                "会做什么",
                "能做",
                "介绍一下",
                "你是谁",
            )
        )
    if len(normalized) <= 12:
        return any(
            token in normalized
            for token in ("能做什么", "会做什么", "介绍一下", "你是谁")
        )
    return False


def _local_whisper_transcribe(
    audio: np.ndarray,
    *,
    sample_rate: int,
    model_name: str,
    cache_dir: str,
    site_packages: str,
    initial_prompt: str,
) -> str:
    global _LOCAL_WHISPER_MODEL, _LOCAL_WHISPER_MODEL_KEY

    if sample_rate != 16000:
        raise ValueError(f"local whisper fallback expects 16kHz audio, got {sample_rate}")
    if site_packages and site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    import contextlib
    import io
    import whisper

    key = (model_name, cache_dir, site_packages)
    with _LOCAL_WHISPER_LOCK:
        if _LOCAL_WHISPER_MODEL is None or _LOCAL_WHISPER_MODEL_KEY != key:
            kwargs = {"download_root": cache_dir} if cache_dir else {}
            _LOCAL_WHISPER_MODEL = whisper.load_model(model_name, **kwargs)
            _LOCAL_WHISPER_MODEL_KEY = key
        model = _LOCAL_WHISPER_MODEL

    flat = np.asarray(audio).reshape(-1)
    if flat.dtype != np.float32:
        flat = np.clip(flat.astype(np.float32), -32768.0, 32767.0) / 32768.0

    # OpenAI whisper's Python package can print tqdm progress even with verbose=False.
    # Keep the voice app logs clean while the tiny fallback runs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = model.transcribe(
            flat,
            language="zh",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt or None,
            temperature=0.0,
            beam_size=1,
            best_of=1,
        )
    return _clean_local_asr_text(str(result.get("text") or ""))


def _workspace_root_for_logs() -> Path | None:
    workspace = os.environ.get("MOSS_WORKSPACE", "").strip()
    if workspace:
        return Path(workspace)
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == ".moss_ws":
            return parent
        candidate = parent / ".moss_ws"
        if candidate.exists():
            return candidate
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


class AsyncAudioInputLoop:
    """
    异步音频输入循环。
    持续从音频输入读取数据，并通过回调发送。
    """

    def __init__(
        self,
        send_callback: Callable[[np.ndarray], None],
        audio_input: AsyncAudioInput,
        *,
        resample_rate: Optional[int] = None,
        frame_duration: Optional[float] = None,
    ):
        self._audio_input = audio_input
        self._resample_rate = resample_rate
        self._frame_duration = frame_duration
        self._send_callback = send_callback
        self._stop_event = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """启动音频循环"""
        if self._loop_task is not None:
            return

        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """停止音频循环"""
        self._stop_event.set()
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None

    async def _run_loop(self) -> None:
        """运行音频循环"""
        try:
            await self._audio_input.start()
            while not self._stop_event.is_set():
                try:
                    # 读取音频数据
                    audio_data = await self._audio_input.read(
                        rate=self._resample_rate,
                        duration=self._frame_duration,
                    )
                    # 通过回调发送
                    self._send_callback(audio_data)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    # 记录错误但继续运行
                    print(f"Error in audio loop: {e}")
                    await asyncio.sleep(0.1)
        finally:
            await self._audio_input.stop()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class AsyncDeafState(AsyncListenerState, AsyncRecognitionCallback):
    """聋状态：忽略所有音频输入"""

    def __init__(self, logger: Optional[LoggerItf] = None):
        self._logger = logger
        self._closed = False

    def name(self) -> AsyncListenerStateName:
        return AsyncListenerStateName.DEAF

    async def start(self) -> None:
        self._closed = False
        if self._logger:
            self._logger.info("Deaf state started")

    async def close(self) -> None:
        self._closed = True
        if self._logger:
            self._logger.info("Deaf state closed")

    async def clear_buffer(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def set_vad(self, vad_time: int) -> None:
        pass

    async def next_state(self) -> Optional[tuple[str, Optional[np.ndarray]]]:
        return None

    # AsyncRecognitionCallback 接口（聋状态忽略所有回调）
    async def on_recognition(self, result: Recognition) -> None:
        pass

    async def on_error(self, error: str) -> None:
        if self._logger:
            self._logger.error(f"Deaf state error: {error}")

    async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        pass


class AsyncListeningState(AsyncListenerState, AsyncRecognitionCallback):
    """
    异步聆听状态。
    持续进行语音识别，支持 VAD 检测。
    """

    def __init__(
        self,
        *,
        recognizer: AsyncRecognizer,
        audio_input: AsyncAudioInput,
        callback: AsyncListenerCallback,
        logger: LoggerItf,
        vad: Optional[Callable[[np.ndarray, Optional[int]], bool]] = None,
        stop_on_sentence: bool = True,
        on_complete_state: Optional[str] = None,
        max_idle_time: float = 10.0,
        on_max_idle_state: Optional[str] = None,
        allow_batch: int = 0,  # <= 0 表示无限
    ):
        self._recognizer = recognizer
        self._audio_input = audio_input
        self._callback = callback
        self._logger = logger
        self._vad = vad
        self._stop_on_sentence = stop_on_sentence
        self._on_complete_state = on_complete_state
        self._max_idle_time = max_idle_time
        self._on_max_idle_state = on_max_idle_state
        self._allow_batch = allow_batch

        self._current_batch: Optional[AsyncRecognitionBatch] = None
        self._audio_queue: deque[np.ndarray] = deque()
        self._audio_loop: Optional[AsyncAudioInputLoop] = None
        self._closed = False
        self._started = False
        self._commit_requested = False
        self._clear_buffer_requested = False
        self._next_state: Optional[tuple[str, Optional[np.ndarray]]] = None
        self._vad_time: Optional[int] = None
        self._last_recognition: Optional[Recognition] = None
        self._ran_batch_count = 0
        self._last_activity_time = time.time()

    def name(self) -> AsyncListenerStateName:
        return AsyncListenerStateName.LISTENING

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._closed = False
        self._logger.info("AsyncListeningState started")

        # 启动主循环
        asyncio.create_task(self._main_loop())

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._started = False

        # 停止音频循环
        if self._audio_loop:
            await self._audio_loop.stop()

        # 关闭当前批次
        if self._current_batch:
            await self._current_batch.close()

        self._logger.info("AsyncListeningState closed")

    async def clear_buffer(self) -> None:
        self._clear_buffer_requested = True
        if self._current_batch:
            await self._current_batch.close()
            self._current_batch = None

    async def commit(self) -> None:
        self._commit_requested = True
        if self._current_batch:
            await self._current_batch.commit()

    async def set_vad(self, vad_time: int) -> None:
        self._vad_time = vad_time

    async def next_state(self) -> Optional[tuple[str, Optional[np.ndarray]]]:
        return self._next_state

    # AsyncRecognitionCallback 接口
    async def on_recognition(self, result: Recognition) -> None:
        if self._closed:
            return

        self._last_recognition = result
        self._last_activity_time = time.time()
        await self._callback.on_recognition(result)

    async def on_error(self, error: str) -> None:
        if self._closed:
            return
        await self._callback.on_error(error)

    async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        if self._closed:
            return
        await self._callback.save_batch(rec, audio)

    async def _main_loop(self) -> None:
        """主状态循环"""
        try:
            while not self._closed and self._should_continue_batches():
                batch_id = uuid()
                self._logger.info(f"Starting ASR batch {batch_id}")

                try:
                    await self._run_asr_batch(batch_id)
                    self._ran_batch_count += 1
                except Exception as e:
                    self._logger.exception(f"Error in ASR batch {batch_id}: {e}")
                    await self._callback.on_error(f"ASR batch error: {e}")

                # 检查是否需要切换到其他状态
                if self._check_idle_timeout():
                    if self._on_max_idle_state:
                        self._next_state = (self._on_max_idle_state, None)
                        break

            self._logger.info("AsyncListeningState main loop finished")

        except asyncio.CancelledError:
            self._logger.info("AsyncListeningState main loop cancelled")
        except Exception as e:
            self._logger.exception(f"Error in main loop: {e}")
            await self._callback.on_error(f"Main loop error: {e}")
        finally:
            self._closed = True

    def _should_continue_batches(self) -> bool:
        """检查是否应该继续运行批次"""
        if self._closed:
            return False
        if self._allow_batch > 0 and self._ran_batch_count >= self._allow_batch:
            return False
        return True

    def _check_idle_timeout(self) -> bool:
        """检查是否空闲超时"""
        idle_time = time.time() - self._last_activity_time
        return idle_time > self._max_idle_time

    async def _run_asr_batch(self, batch_id: str) -> None:
        """运行单个 ASR 批次"""
        # 创建音频队列
        audio_queue: deque[np.ndarray] = deque()

        # 创建音频循环
        self._audio_loop = AsyncAudioInputLoop(
            send_callback=audio_queue.append,
            audio_input=self._audio_input,
            resample_rate=self._recognizer.sample_rate,
            frame_duration=self._recognizer.frame_duration,
        )

        # 创建 ASR 批次
        self._current_batch = await self._recognizer.new_batch(
            callback=self,
            batch_id=batch_id,
            vad=self._vad_time,
            stop_on_sentence=self._stop_on_sentence,
        )

        try:
            # 启动音频循环和 ASR 批次
            await self._audio_loop.start()
            await self._current_batch.start()

            # 处理音频数据
            await self._process_audio_batch(audio_queue)

        finally:
            # 清理资源
            if self._current_batch:
                await self._current_batch.close()
                self._current_batch = None

            if self._audio_loop:
                await self._audio_loop.stop()
                self._audio_loop = None

            self._commit_requested = False
            self._clear_buffer_requested = False
            self._logger.info(f"ASR batch {batch_id} finished")

    async def _process_audio_batch(self, audio_queue: deque[np.ndarray]) -> None:
        """处理音频批次"""
        committed = False
        commit_timeout = Timeleft(0.3)

        while not self._closed:
            # 检查批次是否完成
            if self._current_batch and await self._current_batch.is_done():
                self._logger.info("ASR batch completed")
                break

            # 检查是否需要清空缓冲
            if self._clear_buffer_requested:
                self._logger.info("Clear buffer requested")
                self._clear_buffer_requested = False
                break

            # 处理提交请求
            if self._commit_requested and not committed:
                committed = True
                commit_timeout = Timeleft(0.3)
                if self._current_batch:
                    await self._current_batch.commit()
                self._commit_requested = False
                self._logger.info("Commit requested and processed")

            # 处理音频数据
            if audio_queue:
                audio_data = audio_queue.popleft()
                if self._current_batch:
                    await self._current_batch.buffer(audio_data)

                # 检查 VAD
                if self._vad and self._vad(audio_data, self._vad_time):
                    self._commit_requested = True
                    self._logger.info("VAD detected")

            # 检查提交后超时
            if committed and not commit_timeout.alive():
                self._logger.info("Commit timeout reached")
                break

            # 短暂休眠以避免忙等待
            await asyncio.sleep(0.01)


class AsyncPdtListeningState(AsyncListenerState, AsyncRecognitionCallback):
    """
    异步 Push-to-Talk 聆听状态。
    专门为 PTT 模式设计，不继承 ListeningState 以避免设计矛盾。
    """

    def __init__(
        self,
        *,
        recognizer: AsyncRecognizer,
        audio_input: AsyncAudioInput,
        callback: AsyncListenerCallback,
        logger: LoggerItf,
        vad=None,
    ):
        self._recognizer = recognizer
        self._audio_input = audio_input
        self._callback = callback
        self._logger = logger
        self._vad = vad

        self._batch_id = uuid()
        self._current_batch: Optional[AsyncRecognitionBatch] = None
        self._audio_queue: deque[np.ndarray] = deque()
        self._audio_loop: Optional[AsyncAudioInputLoop] = None
        self._closed = False
        self._started = False
        self._committed = False
        self._seq = 0
        self._last_recognition: Optional[Recognition] = None

        self._next_state: Optional[tuple[str, Optional[np.ndarray]]] = None
        self._last_non_empty_recognition_time: float = 0.0
        self._last_non_empty_text: str = ""
        self._last_text_change_time: float = 0.0
        self._commit_reason: str = ""
        self._stable_text_commit_seconds = _float_env("MOSS_ASR_STABLE_TEXT_COMMIT_SECONDS", 0.45)
        self._long_stable_text_commit_seconds = _float_env("MOSS_ASR_LONG_STABLE_TEXT_COMMIT_SECONDS", 1.2)
        self._short_text_max_chars = _int_env("MOSS_ASR_SHORT_TEXT_MAX_CHARS", 8)
        self._stable_punct_commit_seconds = _float_env("MOSS_ASR_STABLE_PUNCT_COMMIT_SECONDS", 0.25)
        self._empty_text_commit_seconds = _float_env("MOSS_ASR_EMPTY_TEXT_COMMIT_SECONDS", 0.75)
        self._long_empty_text_commit_seconds = _float_env(
            "MOSS_ASR_LONG_EMPTY_TEXT_COMMIT_SECONDS",
            max(self._empty_text_commit_seconds, self._long_stable_text_commit_seconds),
        )
        self._audio_idle_commit_seconds = _float_env("MOSS_ASR_AUDIO_IDLE_COMMIT_SECONDS", 0.8)
        self._final_wait_seconds = _float_env("MOSS_ASR_FINAL_WAIT_SECONDS", 0.35)
        self._empty_final_wait_seconds = _float_env("MOSS_ASR_EMPTY_FINAL_WAIT_SECONDS", 0.35)
        self._server_vad_ms = _int_env("MOSS_ASR_SERVER_VAD_MS", 900)
        self._stable_text_min_quiet_seconds = _float_env("MOSS_ASR_STABLE_TEXT_MIN_QUIET_SECONDS", 0.45)
        self._stable_punct_min_quiet_seconds = _float_env(
            "MOSS_ASR_STABLE_PUNCT_MIN_QUIET_SECONDS",
            min(self._stable_text_min_quiet_seconds, 0.25),
        )
        self._stable_short_min_quiet_seconds = _float_env(
            "MOSS_ASR_STABLE_SHORT_MIN_QUIET_SECONDS",
            min(self._stable_text_min_quiet_seconds, 0.35),
        )
        self._long_empty_text_min_quiet_seconds = _float_env(
            "MOSS_ASR_LONG_EMPTY_TEXT_MIN_QUIET_SECONDS",
            max(self._stable_text_min_quiet_seconds, self._long_stable_text_commit_seconds),
        )
        self._prespeech_batch_max_seconds = _float_env("MOSS_ASR_PRESPEECH_BATCH_MAX_SECONDS", 3.0)
        self._speech_no_text_max_seconds = _float_env("MOSS_ASR_SPEECH_NO_TEXT_MAX_SECONDS", 5.5)
        self._speech_no_text_min_quiet_seconds = _float_env("MOSS_ASR_SPEECH_NO_TEXT_MIN_QUIET_SECONDS", 0.65)
        self._speech_no_text_hard_multiplier = _float_env("MOSS_ASR_SPEECH_NO_TEXT_HARD_MULTIPLIER", 1.35)
        self._speech_no_text_action = os.environ.get("MOSS_ASR_SPEECH_NO_TEXT_ACTION", "rotate").strip().lower()
        self._empty_retry_enabled = _bool_env("MOSS_ASR_EMPTY_RETRY_ENABLED", False)
        self._empty_retry_min_rms = _float_env("MOSS_ASR_EMPTY_RETRY_MIN_RMS", 1200.0)
        self._empty_retry_timeout_seconds = _float_env("MOSS_ASR_EMPTY_RETRY_TIMEOUT_SECONDS", 1.6)
        self._input_noise_gate_enabled = _bool_env("MOSS_ASR_INPUT_NOISE_GATE_ENABLED", True)
        self._input_gate_rms = _float_env("MOSS_ASR_INPUT_GATE_RMS", 450.0)
        self._input_gate_preroll_seconds = _float_env("MOSS_ASR_INPUT_GATE_PREROLL_SECONDS", 0.25)
        self._input_gate_tail_seconds = _float_env("MOSS_ASR_INPUT_GATE_TAIL_SECONDS", 0.55)
        self._input_gate_open_frames = max(1, _int_env("MOSS_ASR_INPUT_GATE_OPEN_FRAMES", 1))
        self._no_text_cooldown_seconds = _float_env("MOSS_ASR_NO_TEXT_COOLDOWN_SECONDS", 0.0)
        self._no_text_cooldown_factor = max(1.0, _float_env("MOSS_ASR_NO_TEXT_COOLDOWN_FACTOR", 1.0))
        self._no_text_cooldown_max_seconds = _float_env(
            "MOSS_ASR_NO_TEXT_COOLDOWN_MAX_SECONDS",
            self._no_text_cooldown_seconds,
        )
        self._local_fallback_enabled = _bool_env("MOSS_ASR_LOCAL_FALLBACK_ENABLED", False)
        self._local_fallback_site_packages = os.environ.get("MOSS_ASR_LOCAL_FALLBACK_SITE_PACKAGES", "").strip()
        self._local_fallback_model = os.environ.get("MOSS_ASR_LOCAL_FALLBACK_MODEL", "tiny").strip() or "tiny"
        self._local_fallback_cache_dir = os.environ.get("MOSS_ASR_LOCAL_FALLBACK_CACHE_DIR", "").strip()
        self._local_fallback_initial_prompt = os.environ.get("MOSS_ASR_LOCAL_FALLBACK_INITIAL_PROMPT", "").strip()
        self._local_fallback_rescue_prompt = os.environ.get("MOSS_ASR_LOCAL_FALLBACK_RESCUE_PROMPT", "").strip()
        self._local_fallback_rescue_model = os.environ.get("MOSS_ASR_LOCAL_FALLBACK_RESCUE_MODEL", "").strip()
        self._local_fallback_min_rms = _float_env("MOSS_ASR_LOCAL_FALLBACK_MIN_RMS", 900.0)
        self._local_fallback_min_seconds = _float_env("MOSS_ASR_LOCAL_FALLBACK_MIN_SECONDS", 0.8)
        self._local_fallback_quiet_seconds = _float_env("MOSS_ASR_LOCAL_FALLBACK_QUIET_SECONDS", 0.45)
        self._local_fallback_timeout_seconds = _float_env("MOSS_ASR_LOCAL_FALLBACK_TIMEOUT_SECONDS", 2.0)
        self._local_fallback_trigger_max_seconds = _float_env("MOSS_ASR_LOCAL_FALLBACK_TRIGGER_MAX_SECONDS", 2.0)
        self._local_fallback_max_audio_seconds = _float_env("MOSS_ASR_LOCAL_FALLBACK_MAX_AUDIO_SECONDS", 6.0)
        self._local_fallback_pad_seconds = _float_env("MOSS_ASR_LOCAL_FALLBACK_PAD_SECONDS", 0.15)
        self._local_fallback_drop_unsafe = _bool_env("MOSS_ASR_LOCAL_FALLBACK_DROP_UNSAFE", False)
        self._save_safe_local_fallback_audio = _bool_env("MOSS_ASR_SAVE_SAFE_LOCAL_FALLBACK_AUDIO", False)
        self._batch_started_at = 0.0
        self._committed_at = 0.0

    def name(self) -> AsyncListenerStateName:
        return AsyncListenerStateName.PDT_LISTENING

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._closed = False
        self._committed = False
        self._seq = 0
        self._last_non_empty_recognition_time = 0.0
        self._last_non_empty_text = ""
        self._last_text_change_time = 0.0
        self._commit_reason = ""
        self._batch_started_at = time.time()
        self._committed_at = 0.0

        try:
            # 发送初始空识别结果（保持与现有行为兼容）
            await self._callback.on_recognition(Recognition(
                batch_id=self._batch_id,
                text="",
                seq=0,
                sentence=False,
                is_last=False,
                created=time.time(),
            ))
        except Exception as e:
            self._logger.warning(f"Failed to send initial recognition in PTT listening state: {e}")
            # 继续启动，不阻止状态启动

        # 启动主循环
        asyncio.create_task(self._main_loop())
        self._logger.info("AsyncPdtListeningState started")

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._started = False

        # 停止音频循环
        if self._audio_loop:
            await self._audio_loop.stop()

        # 关闭当前批次
        if self._current_batch:
            await self._current_batch.close()

        self._logger.info("AsyncPdtListeningState closed")

    async def clear_buffer(self) -> None:
        self._closed = True
        self._next_state = (AsyncListenerStateName.PDT_WAITING.value, None)
        if self._current_batch:
            await self._current_batch.close()

    async def commit(self) -> None:
        """PTT 提交：幂等操作"""
        if self._committed:
            return

        self._committed = True
        self._logger.info("PTT commit requested")

        # 如果还没有识别结果，直接关闭
        if self._last_recognition is None or self._last_recognition.is_last:
            self._closed = True
            return

        # 提交当前批次
        if self._current_batch:
            await self._current_batch.commit()

    async def set_vad(self, vad_time: int) -> None:
        # PTT 模式忽略 VAD
        pass

    async def next_state(self) -> Optional[tuple[str, Optional[np.ndarray]]]:
        if self._next_state:
            return self._next_state

        # 默认切换到 PDT_WAITING 状态
        if self._closed:
            return (AsyncListenerStateName.PDT_WAITING.value, None)

        return None

    # AsyncRecognitionCallback 接口
    async def on_recognition(self, result: Recognition) -> None:
        global _NO_TEXT_FAILURE_COUNT
        # 更新批次 ID 和序列号
        result.batch_id = self._batch_id
        self._seq += 1
        result.seq = self._seq

        self._last_recognition = result
        # 跟踪最后一条非空识别结果
        if result.text and result.text.strip():
            _NO_TEXT_FAILURE_COUNT = 0
            now = time.time()
            text = result.text.strip()
            self._last_non_empty_recognition_time = now
            if text != self._last_non_empty_text:
                self._last_non_empty_text = text
                self._last_text_change_time = now

        # 如果是最后一条结果但文本为空，用最后一条非空文本替换
        if result.is_last and not result.text and self._last_non_empty_text:
            self._logger.info(
                f"PTT final result is empty, using last non-empty text: '{self._last_non_empty_text}'"
            )
            result.text = self._last_non_empty_text

        # 最后一条结果附加提交原因
        if result.is_last:
            result.commit_reason = self._commit_reason or "manual"

        self._logger.info(
            f"PTT on_recognition: text='{result.text}', sentence={result.sentence}, "
            f"is_last={result.is_last}, committed={self._committed}"
        )

        # 非最终结果且已关闭，跳过回调
        if self._closed and not result.is_last:
            return

        await self._callback.on_recognition(result)

        # 如果是最后一条结果，标记为关闭
        if result.is_last:
            self._closed = True

    async def on_error(self, error: str) -> None:
        if self._closed:
            return
        await self._callback.on_error(error)

    async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        if self._closed:
            return
        await self._callback.save_batch(rec, audio)

    async def _main_loop(self) -> None:
        """PTT 主循环"""
        try:
            # 重置 VAD 状态
            if self._vad is not None:
                self._vad.reset()

            # 创建音频队列
            audio_queue: deque[np.ndarray] = deque()

            # 创建音频循环
            self._audio_loop = AsyncAudioInputLoop(
                send_callback=audio_queue.append,
                audio_input=self._audio_input,
                resample_rate=self._recognizer.sample_rate,
                frame_duration=self._recognizer.frame_duration,
            )

            self._logger.warning(
                "[ReachyLatency] asr_tuning energy_hold=%s speech_rms=%s silence_rms=%s "
                "short_stable=%.2fs long_stable=%.2fs "
                "short_max=%d punct=%.2fs empty=%.2fs long_empty=%.2fs audio_idle=%.2fs final_wait=%.2fs "
                "empty_final_wait=%.2fs server_vad=%dms stable_quiet=%.2fs "
                "punct_quiet=%.2fs short_quiet=%.2fs long_empty_quiet=%.2fs prespeech_max=%.2fs "
                "speech_no_text_max=%.2fs speech_no_text_quiet=%.2fs "
                "speech_no_text_action=%s hard=%.2fx empty_retry=%s retry_min_rms=%.1f "
                "input_gate=%s gate_rms=%.1f gate_pre=%.2fs gate_tail=%.2fs "
                "gate_open_frames=%d no_text_cooldown=%.2fs factor=%.2f max=%.2fs "
                "local_fallback=%s local_model=%s local_min_rms=%.1f local_quiet=%.2fs drop_unsafe=%s",
                getattr(self._vad, "_silence_hold_time", None),
                getattr(self._vad, "_speech_threshold", None),
                getattr(self._vad, "_silence_threshold", None),
                self._stable_text_commit_seconds,
                self._long_stable_text_commit_seconds,
                self._short_text_max_chars,
                self._stable_punct_commit_seconds,
                self._empty_text_commit_seconds,
                self._long_empty_text_commit_seconds,
                self._audio_idle_commit_seconds,
                self._final_wait_seconds,
                self._empty_final_wait_seconds,
                self._server_vad_ms,
                self._stable_text_min_quiet_seconds,
                self._stable_punct_min_quiet_seconds,
                self._stable_short_min_quiet_seconds,
                self._long_empty_text_min_quiet_seconds,
                self._prespeech_batch_max_seconds,
                self._speech_no_text_max_seconds,
                self._speech_no_text_min_quiet_seconds,
                self._speech_no_text_action,
                self._speech_no_text_hard_multiplier,
                self._empty_retry_enabled,
                self._empty_retry_min_rms,
                self._input_noise_gate_enabled,
                self._input_gate_rms,
                self._input_gate_preroll_seconds,
                self._input_gate_tail_seconds,
                self._input_gate_open_frames,
                self._no_text_cooldown_seconds,
                self._no_text_cooldown_factor,
                self._no_text_cooldown_max_seconds,
                self._local_fallback_enabled,
                self._local_fallback_model,
                self._local_fallback_min_rms,
                self._local_fallback_quiet_seconds,
                self._local_fallback_drop_unsafe,
            )
            _latency_log(
                "asr_tuning",
                energy_hold=getattr(self._vad, "_silence_hold_time", None),
                speech_rms=getattr(self._vad, "_speech_threshold", None),
                silence_rms=getattr(self._vad, "_silence_threshold", None),
                short_stable=self._stable_text_commit_seconds,
                long_stable=self._long_stable_text_commit_seconds,
                short_max=self._short_text_max_chars,
                punct=self._stable_punct_commit_seconds,
                empty=self._empty_text_commit_seconds,
                long_empty=self._long_empty_text_commit_seconds,
                audio_idle=self._audio_idle_commit_seconds,
                final_wait=self._final_wait_seconds,
                empty_final_wait=self._empty_final_wait_seconds,
                server_vad_ms=self._server_vad_ms,
                stable_quiet=self._stable_text_min_quiet_seconds,
                stable_punct_quiet=self._stable_punct_min_quiet_seconds,
                stable_short_quiet=self._stable_short_min_quiet_seconds,
                long_empty_quiet=self._long_empty_text_min_quiet_seconds,
                prespeech_max=self._prespeech_batch_max_seconds,
                speech_no_text_max=self._speech_no_text_max_seconds,
                speech_no_text_quiet=self._speech_no_text_min_quiet_seconds,
                speech_no_text_action=self._speech_no_text_action,
                speech_no_text_hard_multiplier=self._speech_no_text_hard_multiplier,
                empty_retry=self._empty_retry_enabled,
                empty_retry_min_rms=self._empty_retry_min_rms,
                input_gate=self._input_noise_gate_enabled,
                input_gate_rms=self._input_gate_rms,
                input_gate_preroll=self._input_gate_preroll_seconds,
                input_gate_tail=self._input_gate_tail_seconds,
                input_gate_open_frames=self._input_gate_open_frames,
                no_text_cooldown=self._no_text_cooldown_seconds,
                no_text_cooldown_factor=self._no_text_cooldown_factor,
                no_text_cooldown_max=self._no_text_cooldown_max_seconds,
                local_fallback=self._local_fallback_enabled,
                local_fallback_model=self._local_fallback_model,
                local_fallback_min_rms=self._local_fallback_min_rms,
                local_fallback_min_seconds=self._local_fallback_min_seconds,
                local_fallback_quiet=self._local_fallback_quiet_seconds,
                local_fallback_timeout=self._local_fallback_timeout_seconds,
                local_fallback_trigger_max=self._local_fallback_trigger_max_seconds,
                local_fallback_drop_unsafe=self._local_fallback_drop_unsafe,
                local_fallback_rescue=bool(self._local_fallback_rescue_prompt),
                local_fallback_rescue_model=self._local_fallback_rescue_model,
            )

            # 创建 ASR 批次（启用服务端 VAD 作为备份，不按句停止）
            # stop_on_sentence=False：ASR 不会在句子边界自动结束，
            # 由本地 VAD 检测静音后 commit，服务端 VAD 作为兜底
            self._current_batch = await self._recognizer.new_batch(
                callback=self,
                batch_id=self._batch_id,
                vad=self._server_vad_ms,
                stop_on_sentence=False,
            )

            try:
                # 启动音频循环和 ASR 批次
                await self._audio_loop.start()
                await self._current_batch.start()

                # 处理音频数据
                await self._process_audio_batch(audio_queue)

            finally:
                # 清理资源
                if self._current_batch:
                    await self._current_batch.close()
                    self._current_batch = None

                if self._audio_loop:
                    await self._audio_loop.stop()
                    self._audio_loop = None

                # 设置下一个状态
                self._next_state = (AsyncListenerStateName.PDT_WAITING.value, None)
                self._closed = True

            self._logger.info("AsyncPdtListeningState main loop finished")

        except asyncio.CancelledError:
            self._logger.info("AsyncPdtListeningState main loop cancelled")
        except Exception as e:
            self._logger.exception(f"Error in PTT main loop: {e}")
            await self._callback.on_error(f"PTT main loop error: {e}")
        finally:
            self._closed = True

    async def _do_auto_commit(self, reason: str, **log_fields) -> None:
        """统一的自动提交入口，幂等操作"""
        if self._committed:
            return
        self._committed = True
        self._commit_reason = reason
        self._committed_at = time.time()
        self._logger.warning(
            "[ReachyLatency] asr_auto_commit reason=%s elapsed=%.2fs stable_text=%r",
            reason,
            self._committed_at - self._batch_started_at if self._batch_started_at else 0.0,
            self._last_non_empty_text[:80],
        )
        _latency_log(
            "asr_auto_commit",
            reason=reason,
            elapsed=round(self._committed_at - self._batch_started_at, 3) if self._batch_started_at else 0.0,
            text_len=len(self._last_non_empty_text.strip()),
            **log_fields,
        )
        if self._current_batch:
            await self._current_batch.commit()

    def _empty_text_commit_policy(self, text: str) -> tuple[float, str, float]:
        stripped = text.strip()
        if len(stripped) <= self._short_text_max_chars:
            if _ends_terminal_punctuation(stripped):
                return self._empty_text_commit_seconds, "punct", self._stable_punct_min_quiet_seconds
            return self._empty_text_commit_seconds, "short", self._stable_short_min_quiet_seconds
        return self._long_empty_text_commit_seconds, "long", self._long_empty_text_min_quiet_seconds

    def _local_fallback_empty_text_window(
        self,
        *,
        has_speech: bool,
        local_fallback_attempted: bool,
        max_rms: float,
        first_loud_audio_time: float,
        last_loud_audio_time: float,
    ) -> tuple[bool, bool, float, float]:
        """Return whether local fallback may still rescue an empty short utterance."""
        if (
            not self._local_fallback_enabled
            or local_fallback_attempted
            or not has_speech
            or self._last_non_empty_text.strip()
            or max_rms < self._local_fallback_min_rms
            or self._batch_started_at <= 0
            or last_loud_audio_time <= 0
        ):
            return False, False, 0.0, 0.0
        now = time.time()
        speech_started_at = first_loud_audio_time or self._batch_started_at
        elapsed = now - speech_started_at
        last_loud_age = now - last_loud_audio_time
        if self._local_fallback_trigger_max_seconds > 0 and elapsed > self._local_fallback_trigger_max_seconds:
            return False, False, elapsed, last_loud_age
        ready = (
            elapsed >= self._local_fallback_min_seconds
            and last_loud_age >= self._local_fallback_quiet_seconds
        )
        return True, ready, elapsed, last_loud_age

    async def _process_audio_batch(self, audio_queue: deque[np.ndarray]) -> None:
        """处理 PTT 音频批次"""
        global _NO_TEXT_COOLDOWN_UNTIL, _NO_TEXT_FAILURE_COUNT
        self._logger.info("PTT _process_audio_batch started")
        chunk_count = 0
        last_audio_time = time.time()  # 最后一次收到音频的时间
        last_loud_audio_time = 0.0
        first_loud_audio_time = 0.0
        has_speech = False  # 是否检测到过真正超过阈值的语音活动
        max_rms = 0.0
        speech_threshold = float(getattr(self._vad, "_speech_threshold", 600.0) or 600.0)
        gate_threshold = max(0.0, self._input_gate_rms)
        activity_threshold = max(speech_threshold, gate_threshold) if self._input_noise_gate_enabled else speech_threshold
        frame_duration = float(getattr(self._recognizer, "frame_duration", 0.1) or 0.1)
        preroll_frames = max(1, int(self._input_gate_preroll_seconds / frame_duration))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        input_gate_open = not self._input_noise_gate_enabled
        input_gate_last_loud_time = 0.0
        input_gate_loud_frames = 0
        cooldown_skip_logged = False
        local_fallback_attempted = False
        while not self._closed:
            # 检查批次是否完成
            if self._current_batch and await self._current_batch.is_done():
                self._logger.info("PTT ASR batch completed (is_done=True)")
                break

            # 处理音频数据
            if audio_queue:
                audio_data = audio_queue.popleft()
                last_audio_time = time.time()
                rms = float(np.sqrt(np.mean(audio_data.astype(float) ** 2)))
                max_rms = max(max_rms, rms)
                if self._no_text_cooldown_seconds > 0:
                    cooldown_remaining = _NO_TEXT_COOLDOWN_UNTIL - last_audio_time
                    if cooldown_remaining > 0:
                        if not cooldown_skip_logged:
                            cooldown_skip_logged = True
                            _latency_log(
                                "asr_no_text_cooldown_skip",
                                remaining=round(cooldown_remaining, 3),
                                rms=round(rms, 1),
                                max_rms=round(max_rms, 1),
                            )
                        continue
                    cooldown_skip_logged = False
                should_send_audio = True
                if self._input_noise_gate_enabled:
                    should_send_audio = False
                    preroll.append(audio_data)
                    if rms >= gate_threshold:
                        input_gate_last_loud_time = last_audio_time
                        input_gate_loud_frames += 1
                        if not input_gate_open:
                            if input_gate_loud_frames >= self._input_gate_open_frames:
                                input_gate_open = True
                                _latency_log(
                                    "asr_input_gate_open",
                                    rms=round(rms, 1),
                                    threshold=round(gate_threshold, 1),
                                    preroll_frames=len(preroll),
                                    confirmed_frames=input_gate_loud_frames,
                                )
                                if self._current_batch:
                                    for frame in list(preroll)[:-1]:
                                        await self._current_batch.buffer(frame)
                        should_send_audio = input_gate_open
                    elif input_gate_open and input_gate_last_loud_time > 0:
                        input_gate_loud_frames = 0
                        should_send_audio = (
                            last_audio_time - input_gate_last_loud_time
                            <= self._input_gate_tail_seconds
                        )
                        if not should_send_audio:
                            input_gate_open = False
                            _latency_log(
                                "asr_input_gate_close",
                                quiet=round(last_audio_time - input_gate_last_loud_time, 3),
                            )
                            preroll.clear()
                    else:
                        input_gate_loud_frames = 0
                        should_send_audio = False
                if should_send_audio and rms >= activity_threshold:
                    if not has_speech:
                        first_loud_audio_time = last_audio_time
                    last_loud_audio_time = last_audio_time
                    has_speech = True
                if should_send_audio and self._current_batch:
                    await self._current_batch.buffer(audio_data)

                # 本地 VAD 静音检测
                if self._vad is not None and not self._committed and should_send_audio:
                    chunk_count += 1
                    should_commit = self._vad(audio_data)
                    # 每50个chunk打印一次诊断信息（约2.5秒）
                    if chunk_count % 50 == 0:
                        self._logger.info(
                            f"PTT VAD diag: chunk={chunk_count}, rms={rms:.1f}, "
                            f"speech_detected={self._vad._speech_detected}, "
                            f"silence_start={self._vad._silence_start}, "
                            f"committed={self._committed}"
                        )
                    if should_commit:
                        self._logger.info(
                            f"VAD detected silence after speech, auto-committing (rms={rms:.1f})"
                        )
                        fallback_eligible, fallback_ready, fallback_elapsed, fallback_last_loud_age = (
                            self._local_fallback_empty_text_window(
                                has_speech=has_speech,
                                local_fallback_attempted=local_fallback_attempted,
                                max_rms=max_rms,
                                first_loud_audio_time=first_loud_audio_time,
                                last_loud_audio_time=last_loud_audio_time,
                            )
                        )
                        if fallback_eligible:
                            _latency_log(
                                "asr_energy_vad_defer_for_local_fallback",
                                ready=fallback_ready,
                                speech_elapsed=round(fallback_elapsed, 3),
                                last_loud_age=round(fallback_last_loud_age, 3),
                                max_rms=round(max_rms, 1),
                            )
                        else:
                            await self._do_auto_commit(
                                "energy_vad",
                                max_rms=round(max_rms, 1),
                                has_speech=has_speech,
                                last_loud_age=round(time.time() - last_loud_audio_time, 3)
                                if last_loud_audio_time > 0
                                else None,
                            )

            # 没听到明确人声时定期刷新 ASR 批次，避免短句落入长时间静音流后被服务端返回空文本。
            if (
                not self._committed
                and not has_speech
                and self._prespeech_batch_max_seconds > 0
                and self._batch_started_at > 0
                and time.time() - self._batch_started_at >= self._prespeech_batch_max_seconds
            ):
                elapsed = time.time() - self._batch_started_at
                self._logger.info(
                    "No speech before %.1fs, rotating ASR batch (max_rms=%.1f)",
                    elapsed,
                    max_rms,
                )
                _latency_log(
                    "asr_prespeech_rotate",
                    elapsed=round(elapsed, 3),
                    max_rms=round(max_rms, 1),
                )
                break

            # 短句听到了能量但云端还没出字时，用本地 tiny whisper 抢救一次。
            # 只在批次早期触发，避免长句中途停顿被提前截断。
            if (
                not self._committed
                and self._local_fallback_enabled
                and not local_fallback_attempted
                and has_speech
                and not self._last_non_empty_text.strip()
                and max_rms >= self._local_fallback_min_rms
                and self._batch_started_at > 0
                and last_loud_audio_time > 0
            ):
                speech_started_at = first_loud_audio_time or self._batch_started_at
                elapsed = time.time() - speech_started_at
                last_loud_age = time.time() - last_loud_audio_time
                if (
                    elapsed >= self._local_fallback_min_seconds
                    and last_loud_age >= self._local_fallback_quiet_seconds
                    and (
                        self._local_fallback_trigger_max_seconds <= 0
                        or elapsed <= self._local_fallback_trigger_max_seconds
                    )
                ):
                    local_fallback_attempted = True
                    fallback_text = await self._try_local_fallback(
                        reason="local_whisper_quiet",
                        max_rms=max_rms,
                        last_loud_age=last_loud_age,
                    )
                    if fallback_text:
                        _NO_TEXT_FAILURE_COUNT = 0
                        await self._finish_with_local_fallback(
                            fallback_text,
                            reason="local_whisper_quiet",
                            max_rms=max_rms,
                            last_loud_age=last_loud_age,
                        )
                        break
                    if self._local_fallback_drop_unsafe:
                        self._start_no_text_cooldown(
                            reason="local_fallback_no_safe_text",
                            max_rms=max_rms,
                        )
                        _latency_log(
                            "asr_local_fallback_noise_drop",
                            max_rms=round(max_rms, 1),
                            last_loud_age=round(last_loud_age, 3),
                            speech_elapsed=round(elapsed, 3),
                            elapsed=round(time.time() - self._batch_started_at, 3)
                            if self._batch_started_at
                            else 0.0,
                        )
                        await self._save_debug_batch(
                            reason="local_fallback_no_safe_text",
                            max_rms=max_rms,
                        )
                        break
                    # Local fallback may block for hundreds of ms. Do not let that
                    # pause look like input silence while captured audio is queued.
                    last_audio_time = time.time()
                    if audio_queue:
                        continue

            # 如果本地 VAD 明确听到过声音，但云端 ASR 长时间没有任何文字，
            # 这通常是被背景噪声或批次边界卡住的坏流。尽快旋转，避免短唤醒词被旧流吞掉。
            if (
                not self._committed
                and has_speech
                and not self._last_non_empty_text.strip()
                and self._speech_no_text_max_seconds > 0
                and self._batch_started_at > 0
            ):
                speech_started_at = first_loud_audio_time or self._batch_started_at
                elapsed = time.time() - speech_started_at
                if elapsed >= self._speech_no_text_max_seconds:
                    last_loud_age = (
                        time.time() - last_loud_audio_time
                        if last_loud_audio_time > 0
                        else float("inf")
                    )
                    hard_elapsed = self._speech_no_text_max_seconds * self._speech_no_text_hard_multiplier
                    if last_loud_age >= self._speech_no_text_min_quiet_seconds or elapsed >= hard_elapsed:
                        reason = "speech_no_text_timeout"
                        if self._speech_no_text_action in {"rotate", "reset", "drop"}:
                            self._commit_reason = reason
                            self._logger.info(
                                "Speech energy without ASR text for %.1fs "
                                "(max_rms=%.1f, last_loud_age=%.1f), rotating batch",
                                elapsed,
                                max_rms,
                                last_loud_age,
                            )
                            _latency_log(
                                "asr_speech_no_text_rotate",
                                elapsed=round(elapsed, 3),
                                batch_elapsed=round(time.time() - self._batch_started_at, 3)
                                if self._batch_started_at
                                else 0.0,
                                max_rms=round(max_rms, 1),
                                last_loud_age=round(last_loud_age, 3)
                                if last_loud_age != float("inf")
                                else None,
                                hard_elapsed=round(hard_elapsed, 3),
                            )
                            self._start_no_text_cooldown(
                                reason=reason,
                                max_rms=max_rms,
                            )
                            await self._save_debug_batch(reason=reason, max_rms=max_rms)
                            break
                        self._logger.info(
                            "Speech energy without ASR text for %.1fs "
                            "(max_rms=%.1f, last_loud_age=%.1f), auto-committing",
                            elapsed,
                            max_rms,
                            last_loud_age,
                        )
                        await self._do_auto_commit(
                            reason,
                            max_rms=round(max_rms, 1),
                            last_loud_age=round(last_loud_age, 3)
                            if last_loud_age != float("inf")
                            else None,
                        )

            # 音频队列空闲检测：如果检测到过语音活动，且音频队列持续为空超过配置时间，自动提交
            # 这个检测不依赖 ASR 返回空文本，直接基于音频输入
            if not self._committed and has_speech and not audio_queue:
                audio_idle = time.time() - last_audio_time
                if audio_idle >= self._audio_idle_commit_seconds:
                    self._logger.info(
                        f"Audio queue idle for {audio_idle:.1f}s after speech, auto-committing"
                    )
                    await self._do_auto_commit(
                        "audio_idle",
                        max_rms=round(max_rms, 1),
                        last_loud_age=round(time.time() - last_loud_audio_time, 3)
                        if last_loud_audio_time > 0
                        else None,
                    )

            # ASR 文本稳定检测：服务端有时会持续重复同一句 partial，不再继续变化。
            # 这种情况下按稳定时间主动提交，避免短句卡十几秒才进入 LLM。
            if not self._committed and self._last_text_change_time > 0:
                stable_elapsed = time.time() - self._last_text_change_time
                text = self._last_non_empty_text
                if len(text.strip()) <= self._short_text_max_chars and _ends_terminal_punctuation(text):
                    threshold = self._stable_punct_commit_seconds
                    threshold_kind = "punct"
                    min_quiet_seconds = self._stable_punct_min_quiet_seconds
                elif len(text.strip()) <= self._short_text_max_chars:
                    threshold = self._stable_text_commit_seconds
                    threshold_kind = "short"
                    min_quiet_seconds = self._stable_short_min_quiet_seconds
                else:
                    threshold = self._long_stable_text_commit_seconds
                    threshold_kind = "long"
                    min_quiet_seconds = self._stable_text_min_quiet_seconds
                if stable_elapsed >= threshold:
                    quiet_elapsed = (
                        time.time() - last_loud_audio_time
                        if last_loud_audio_time > 0
                        else float("inf")
                    )
                    if quiet_elapsed < min_quiet_seconds:
                        await asyncio.sleep(0.01)
                        continue
                    self._logger.info(
                        "ASR stable text timeout (%.1fs, threshold=%.1fs kind=%s, quiet=%.1fs min_quiet=%.1fs, text=%r), auto-committing",
                        stable_elapsed,
                        threshold,
                        threshold_kind,
                        quiet_elapsed,
                        min_quiet_seconds,
                        text[:80],
                    )
                    await self._do_auto_commit(
                        f"stable_text_{threshold_kind}",
                        quiet=round(quiet_elapsed, 3)
                        if quiet_elapsed != float("inf")
                        else None,
                        min_quiet=round(min_quiet_seconds, 3),
                    )

            # ASR 空文本超时检测：如果已经识别到过文字，且超过配置时间没有新的非空结果，自动提交
            if not self._committed and self._last_non_empty_recognition_time > 0:
                elapsed = time.time() - self._last_non_empty_recognition_time
                empty_threshold, empty_kind, min_quiet_seconds = self._empty_text_commit_policy(
                    self._last_non_empty_text,
                )
                if elapsed >= empty_threshold:
                    quiet_elapsed = (
                        time.time() - last_loud_audio_time
                        if last_loud_audio_time > 0
                        else float("inf")
                    )
                    if quiet_elapsed < min_quiet_seconds:
                        await asyncio.sleep(0.01)
                        continue
                    self._logger.info(
                        "ASR empty text timeout (%.1fs, threshold=%.1fs kind=%s, quiet=%.1fs min_quiet=%.1fs), auto-committing",
                        elapsed,
                        empty_threshold,
                        empty_kind,
                        quiet_elapsed,
                        min_quiet_seconds,
                    )
                    await self._do_auto_commit(
                        "empty_text_timeout",
                        empty_kind=empty_kind,
                        threshold=round(empty_threshold, 3),
                        quiet=round(quiet_elapsed, 3)
                        if quiet_elapsed != float("inf")
                        else None,
                        min_quiet=round(min_quiet_seconds, 3),
                    )

            # 检查是否已提交
            if self._committed:
                # 等待批次完成或超时
                wait_started = time.time()
                final_wait_seconds = (
                    self._empty_final_wait_seconds
                    if not self._last_non_empty_text.strip()
                    else self._final_wait_seconds
                )
                try:
                    await asyncio.wait_for(
                        self._current_batch.wait_until_done(),
                        timeout=final_wait_seconds,
                    )
                except asyncio.TimeoutError:
                    self._logger.warning(
                        "PTT wait_until_done timed out after %.1fs; closing with last recognition",
                        final_wait_seconds,
                    )
                if (
                    self._empty_retry_enabled
                    and not self._last_non_empty_text.strip()
                    and max_rms >= self._empty_retry_min_rms
                    and self._current_batch is not None
                ):
                    audio = await self._current_batch.get_buffer()
                    retry_text = await self._retry_empty_audio(audio, max_rms=max_rms)
                    if retry_text:
                        self._logger.info("ASR empty retry recovered text=%r", retry_text[:80])
                self._logger.warning(
                    "[ReachyLatency] asr_final_wait done elapsed=%.2fs total=%.2fs reason=%s",
                    time.time() - wait_started,
                    time.time() - self._batch_started_at if self._batch_started_at else 0.0,
                    self._commit_reason or "manual",
                )
                _latency_log(
                    "asr_final_wait",
                    elapsed=round(time.time() - wait_started, 3),
                    total=round(time.time() - self._batch_started_at, 3) if self._batch_started_at else 0.0,
                    reason=self._commit_reason or "manual",
                    text_len=len(self._last_non_empty_text.strip()),
                )
                if has_speech and not self._last_non_empty_text.strip() and max_rms > 0:
                    await self._save_debug_batch(
                        reason=self._commit_reason or "empty_speech_commit",
                        max_rms=max_rms,
                    )
                break

            # 短暂休眠以避免忙等待
            await asyncio.sleep(0.01)

    def _start_no_text_cooldown(self, *, reason: str, max_rms: float) -> None:
        global _NO_TEXT_COOLDOWN_UNTIL, _NO_TEXT_FAILURE_COUNT
        if self._no_text_cooldown_seconds <= 0:
            return
        _NO_TEXT_FAILURE_COUNT += 1
        scaled = self._no_text_cooldown_seconds * (
            self._no_text_cooldown_factor ** max(0, _NO_TEXT_FAILURE_COUNT - 1)
        )
        cap = self._no_text_cooldown_max_seconds
        seconds = min(scaled, cap) if cap > 0 else scaled
        _NO_TEXT_COOLDOWN_UNTIL = max(_NO_TEXT_COOLDOWN_UNTIL, time.time() + seconds)
        _latency_log(
            "asr_no_text_cooldown_start",
            seconds=round(seconds, 3),
            base_seconds=round(self._no_text_cooldown_seconds, 3),
            failures=_NO_TEXT_FAILURE_COUNT,
            reason=reason,
            max_rms=round(max_rms, 1),
            until=round(_NO_TEXT_COOLDOWN_UNTIL, 3),
        )

    async def _save_debug_batch(self, *, reason: str, max_rms: float) -> None:
        if self._current_batch is None:
            return
        try:
            audio = await self._current_batch.get_buffer()
        except Exception as e:
            self._logger.debug("Failed to get ASR debug audio: %s", e)
            return
        if audio is None or len(audio) <= 0:
            return
        rec = Recognition(
            batch_id=self._batch_id,
            text=self._last_non_empty_text.strip(),
            seq=self._seq + 1,
            sentence=False,
            is_last=True,
            created=time.time(),
            commit_reason=reason,
            audio_max_rms=round(max_rms, 1),
        )
        _latency_log(
            "asr_debug_batch_save_requested",
            reason=reason,
            samples=int(len(audio)),
            max_rms=round(max_rms, 1),
        )
        await self._callback.save_batch(rec, audio)

    async def _try_local_fallback(self, *, reason: str, max_rms: float, last_loud_age: float) -> str:
        if self._current_batch is None:
            return ""
        try:
            audio = await self._current_batch.get_buffer()
        except Exception as e:
            _latency_log("asr_local_fallback_error", reason=reason, error=str(e)[:160])
            return ""
        if audio is None or len(audio) <= 0:
            return ""

        sample_rate = int(getattr(self._recognizer, "sample_rate", 16000) or 16000)
        flat = np.asarray(audio).reshape(-1)
        if self._local_fallback_max_audio_seconds > 0:
            max_samples = int(sample_rate * self._local_fallback_max_audio_seconds)
            if len(flat) > max_samples:
                flat = flat[-max_samples:]
        if self._local_fallback_pad_seconds > 0:
            pad_samples = int(sample_rate * self._local_fallback_pad_seconds)
            if pad_samples > 0:
                pad = np.zeros(pad_samples, dtype=flat.dtype)
                flat = np.concatenate([pad, flat, pad])

        started = time.time()
        _latency_log(
            "asr_local_fallback_start",
            reason=reason,
            samples=int(len(flat)),
            duration=round(len(flat) / sample_rate, 3) if sample_rate > 0 else 0.0,
            max_rms=round(max_rms, 1),
            last_loud_age=round(last_loud_age, 3),
            model=self._local_fallback_model,
        )
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(
                    _local_whisper_transcribe,
                    flat,
                    sample_rate=sample_rate,
                    model_name=self._local_fallback_model,
                    cache_dir=self._local_fallback_cache_dir,
                    site_packages=self._local_fallback_site_packages,
                    initial_prompt=self._local_fallback_initial_prompt,
                ),
                timeout=max(0.1, self._local_fallback_timeout_seconds),
            )
        except asyncio.TimeoutError:
            _latency_log(
                "asr_local_fallback_timeout",
                reason=reason,
                elapsed=round(time.time() - started, 3),
                timeout=self._local_fallback_timeout_seconds,
            )
            return ""
        except Exception as e:
            _latency_log(
                "asr_local_fallback_error",
                reason=reason,
                elapsed=round(time.time() - started, 3),
                error=str(e)[:200],
            )
            return ""

        _latency_log(
            "asr_local_fallback_done",
            reason=reason,
            elapsed=round(time.time() - started, 3),
            text_len=len(text),
            text_preview=text[:40],
        )
        if (
            text
            and self._local_fallback_rescue_prompt
            and not _is_safe_local_fallback_text(text)
            and _looks_like_rescuable_short_wake_fragment(_normalize_local_asr_text(text))
        ):
            rescue_started = time.time()
            _latency_log(
                "asr_local_fallback_rescue_start",
                reason=reason,
                text_len=len(text),
                text_preview=text[:40],
            )
            try:
                rescue_text = await asyncio.wait_for(
                    asyncio.to_thread(
                        _local_whisper_transcribe,
                        flat,
                        sample_rate=sample_rate,
                        model_name=self._local_fallback_model,
                        cache_dir=self._local_fallback_cache_dir,
                        site_packages=self._local_fallback_site_packages,
                        initial_prompt=self._local_fallback_rescue_prompt,
                    ),
                    timeout=max(0.1, self._local_fallback_timeout_seconds),
                )
            except Exception as e:
                _latency_log(
                    "asr_local_fallback_rescue_error",
                    reason=reason,
                    elapsed=round(time.time() - rescue_started, 3),
                    error=str(e)[:200],
                )
            else:
                _latency_log(
                    "asr_local_fallback_rescue_done",
                    reason=reason,
                    elapsed=round(time.time() - rescue_started, 3),
                    text_len=len(rescue_text),
                    text_preview=rescue_text[:40],
                )
                if rescue_text and _is_safe_local_fallback_text(rescue_text):
                    text = _canonicalize_safe_local_fallback_text(rescue_text)
                else:
                    _latency_log(
                        "asr_local_fallback_rescue_reject",
                        reason=reason,
                        text_len=len(rescue_text),
                        text_preview=rescue_text[:40],
                    )
        if (
            text
            and self._local_fallback_rescue_model
            and self._local_fallback_rescue_model != self._local_fallback_model
            and not _is_safe_local_fallback_text(text)
            and _looks_like_rescuable_wake_second_pass(_normalize_local_asr_text(text))
        ):
            rescue_model_started = time.time()
            _latency_log(
                "asr_local_fallback_rescue_model_start",
                reason=reason,
                model=self._local_fallback_rescue_model,
                text_len=len(text),
                text_preview=text[:40],
            )
            try:
                rescue_model_text = await asyncio.wait_for(
                    asyncio.to_thread(
                        _local_whisper_transcribe,
                        flat,
                        sample_rate=sample_rate,
                        model_name=self._local_fallback_rescue_model,
                        cache_dir=self._local_fallback_cache_dir,
                        site_packages=self._local_fallback_site_packages,
                        initial_prompt=self._local_fallback_rescue_prompt,
                    ),
                    timeout=max(0.1, self._local_fallback_timeout_seconds),
                )
            except Exception as e:
                _latency_log(
                    "asr_local_fallback_rescue_model_error",
                    reason=reason,
                    model=self._local_fallback_rescue_model,
                    elapsed=round(time.time() - rescue_model_started, 3),
                    error=str(e)[:200],
                )
            else:
                _latency_log(
                    "asr_local_fallback_rescue_model_done",
                    reason=reason,
                    model=self._local_fallback_rescue_model,
                    elapsed=round(time.time() - rescue_model_started, 3),
                    text_len=len(rescue_model_text),
                    text_preview=rescue_model_text[:40],
                )
                if rescue_model_text and _is_safe_local_fallback_text(rescue_model_text):
                    text = _canonicalize_safe_local_fallback_text(rescue_model_text)
                    _latency_log(
                        "asr_local_fallback_rescue_model_accept",
                        reason=reason,
                        model=self._local_fallback_rescue_model,
                        text_len=len(text),
                        text_preview=text[:40],
                    )
                else:
                    _latency_log(
                        "asr_local_fallback_rescue_model_reject",
                        reason=reason,
                        model=self._local_fallback_rescue_model,
                        text_len=len(rescue_model_text),
                        text_preview=rescue_model_text[:40],
                    )
        if text and not _is_safe_local_fallback_text(text):
            _latency_log(
                "asr_local_fallback_reject",
                reason=reason,
                text_len=len(text),
                text_preview=text[:40],
                reject_reason="unsafe_text",
            )
            return ""
        return _canonicalize_safe_local_fallback_text(text)

    async def _finish_with_local_fallback(
        self,
        text: str,
        *,
        reason: str,
        max_rms: float,
        last_loud_age: float,
    ) -> None:
        text = _clean_local_asr_text(text)
        if not text:
            return
        self._commit_reason = reason
        self._last_non_empty_text = text
        now = time.time()
        self._last_non_empty_recognition_time = now
        self._last_text_change_time = now
        _latency_log(
            "asr_local_fallback_emit",
            reason=reason,
            text_len=len(text),
            text_preview=text[:40],
            max_rms=round(max_rms, 1),
            last_loud_age=round(last_loud_age, 3),
        )
        rec = Recognition(
            batch_id=self._batch_id,
            text=text,
            seq=self._seq + 1,
            sentence=True,
            is_last=True,
            created=now,
            commit_reason=reason,
            audio_max_rms=round(max_rms, 1),
        )
        if self._save_safe_local_fallback_audio and self._current_batch is not None:
            try:
                audio = await self._current_batch.get_buffer()
            except Exception as e:
                _latency_log("asr_local_fallback_safe_audio_save_error", reason=reason, error=str(e)[:160])
            else:
                if audio is not None and len(audio) > 0:
                    save_rec = Recognition(
                        batch_id=self._batch_id,
                        text=text,
                        seq=self._seq + 1,
                        sentence=True,
                        is_last=True,
                        created=now,
                        commit_reason="local_fallback_safe_text",
                        audio_max_rms=round(max_rms, 1),
                    )
                    _latency_log(
                        "asr_local_fallback_safe_audio_save_requested",
                        reason=reason,
                        samples=int(len(audio)),
                        text_len=len(text),
                        max_rms=round(max_rms, 1),
                    )
                    await self._callback.save_batch(save_rec, audio)
        await self.on_recognition(rec)

    async def _retry_empty_audio(self, audio: np.ndarray, *, max_rms: float) -> str:
        if audio is None or len(audio) <= 0:
            return ""
        sample_rate = int(getattr(self._recognizer, "sample_rate", 16000) or 16000)
        frame_duration = float(getattr(self._recognizer, "frame_duration", 0.1) or 0.1)
        frame_samples = max(1, int(sample_rate * frame_duration))
        retry_started = time.time()
        retry_batch = await self._recognizer.new_batch(
            callback=self,
            batch_id=uuid(),
            vad=self._server_vad_ms,
            stop_on_sentence=False,
        )
        self._logger.info(
            "Retrying empty high-energy ASR batch: samples=%d max_rms=%.1f",
            len(audio),
            max_rms,
        )
        _latency_log(
            "asr_empty_retry_start",
            samples=int(len(audio)),
            max_rms=round(max_rms, 1),
        )
        try:
            await retry_batch.start()
            for pos in range(0, len(audio), frame_samples):
                await retry_batch.buffer(audio[pos : pos + frame_samples])
                await asyncio.sleep(0.002)
            await retry_batch.commit()
            await retry_batch.wait_until_done(timeout=self._empty_retry_timeout_seconds)
        except Exception as e:
            self._logger.warning("ASR empty retry failed: %s", e)
        finally:
            try:
                await retry_batch.close()
            except Exception:
                pass
        text = self._last_non_empty_text.strip()
        _latency_log(
            "asr_empty_retry_done",
            elapsed=round(time.time() - retry_started, 3),
            text_len=len(text),
        )
        return text


class AsyncPdtWaitingState(AsyncListenerState):
    """异步 Push-to-Talk 等待状态"""

    def __init__(self):
        self._next_state: Optional[tuple[str, Optional[np.ndarray]]] = None
        self._closed = False

    def name(self) -> AsyncListenerStateName:
        return AsyncListenerStateName.PDT_WAITING

    async def start(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def clear_buffer(self) -> None:
        pass

    async def commit(self) -> None:
        """PTT 等待状态下的提交：切换到聆听状态"""
        self._next_state = (AsyncListenerStateName.PDT_LISTENING.value, None)

    async def set_vad(self, vad_time: int) -> None:
        pass

    async def next_state(self) -> Optional[tuple[str, Optional[np.ndarray]]]:
        return self._next_state


# 注意：AsyncAsleepState 跳过实现，因为唤醒词功能跳过
# 可以直接使用 AsyncDeafState 代替
