"""
异步 Listener 状态实现。

完全基于 asyncio，弃用线程模型。
"""

import asyncio
import os
import time
from collections import deque
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


def _ends_terminal_punctuation(text: str) -> bool:
    return text.rstrip().endswith(("。", "？", "?", "！", "!", "；", ";", ".", "…"))


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
        self._stable_text_commit_seconds = _float_env("MOSS_ASR_STABLE_TEXT_COMMIT_SECONDS", 1.0)
        self._stable_punct_commit_seconds = _float_env("MOSS_ASR_STABLE_PUNCT_COMMIT_SECONDS", 0.45)
        self._empty_text_commit_seconds = _float_env("MOSS_ASR_EMPTY_TEXT_COMMIT_SECONDS", 1.0)
        self._audio_idle_commit_seconds = _float_env("MOSS_ASR_AUDIO_IDLE_COMMIT_SECONDS", 1.0)
        self._final_wait_seconds = _float_env("MOSS_ASR_FINAL_WAIT_SECONDS", 1.2)

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
        # 更新批次 ID 和序列号
        result.batch_id = self._batch_id
        self._seq += 1
        result.seq = self._seq

        self._last_recognition = result
        # 跟踪最后一条非空识别结果
        if result.text and result.text.strip():
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

            # 创建 ASR 批次（启用服务端 VAD 作为备份，不按句停止）
            # stop_on_sentence=False：ASR 不会在句子边界自动结束，
            # 由本地 VAD 检测静音后 commit，服务端 vad=2000 作为兜底
            self._current_batch = await self._recognizer.new_batch(
                callback=self,
                batch_id=self._batch_id,
                vad=2000,  # 服务端 VAD：2秒静音自动分句
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

    async def _do_auto_commit(self, reason: str) -> None:
        """统一的自动提交入口，幂等操作"""
        if self._committed:
            return
        self._committed = True
        self._commit_reason = reason
        self._logger.info(f"Auto-commit triggered by: {reason}")
        if self._current_batch:
            await self._current_batch.commit()

    async def _process_audio_batch(self, audio_queue: deque[np.ndarray]) -> None:
        """处理 PTT 音频批次"""
        self._logger.info("PTT _process_audio_batch started")
        chunk_count = 0
        last_audio_time = time.time()  # 最后一次收到音频的时间
        has_speech = False  # 是否检测到过语音活动（音频队列非空）
        while not self._closed:
            # 检查批次是否完成
            if self._current_batch and await self._current_batch.is_done():
                self._logger.info("PTT ASR batch completed (is_done=True)")
                break

            # 处理音频数据
            if audio_queue:
                audio_data = audio_queue.popleft()
                last_audio_time = time.time()
                has_speech = True
                if self._current_batch:
                    await self._current_batch.buffer(audio_data)

                # 本地 VAD 静音检测
                if self._vad is not None and not self._committed:
                    chunk_count += 1
                    rms = float(np.sqrt(np.mean(audio_data.astype(float) ** 2)))
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
                        await self._do_auto_commit("energy_vad")

            # 音频队列空闲检测：如果检测到过语音活动，且音频队列持续为空超过配置时间，自动提交
            # 这个检测不依赖 ASR 返回空文本，直接基于音频输入
            if not self._committed and has_speech:
                audio_idle = time.time() - last_audio_time
                if audio_idle >= self._audio_idle_commit_seconds:
                    self._logger.info(
                        f"Audio queue idle for {audio_idle:.1f}s after speech, auto-committing"
                    )
                    await self._do_auto_commit("audio_idle")

            # ASR 文本稳定检测：服务端有时会持续重复同一句 partial，不再继续变化。
            # 这种情况下按稳定时间主动提交，避免短句卡十几秒才进入 LLM。
            if not self._committed and self._last_text_change_time > 0:
                stable_elapsed = time.time() - self._last_text_change_time
                text = self._last_non_empty_text
                threshold = (
                    self._stable_punct_commit_seconds
                    if _ends_terminal_punctuation(text)
                    else self._stable_text_commit_seconds
                )
                if stable_elapsed >= threshold:
                    self._logger.info(
                        "ASR stable text timeout (%.1fs, threshold=%.1fs, text=%r), auto-committing",
                        stable_elapsed,
                        threshold,
                        text[:80],
                    )
                    await self._do_auto_commit("stable_text")

            # ASR 空文本超时检测：如果已经识别到过文字，且超过配置时间没有新的非空结果，自动提交
            if not self._committed and self._last_non_empty_recognition_time > 0:
                elapsed = time.time() - self._last_non_empty_recognition_time
                if elapsed >= self._empty_text_commit_seconds:
                    self._logger.info(
                        f"ASR empty text timeout ({elapsed:.1f}s since last non-empty result), auto-committing"
                    )
                    await self._do_auto_commit("empty_text_timeout")

            # 检查是否已提交
            if self._committed:
                # 等待批次完成或超时
                try:
                    await asyncio.wait_for(
                        self._current_batch.wait_until_done(),
                        timeout=self._final_wait_seconds,
                    )
                except asyncio.TimeoutError:
                    self._logger.warning(
                        "PTT wait_until_done timed out after %.1fs; closing with last recognition",
                        self._final_wait_seconds,
                    )
                break

            # 短暂休眠以避免忙等待
            await asyncio.sleep(0.01)


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
