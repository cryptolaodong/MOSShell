"""
异步火山引擎语音识别实现。

完全基于 asyncio，弃用线程模型。
重用 volcengine_bm_protocol 中的协议函数。
"""

import asyncio
import json
import os
import time
from collections import deque
from typing import Optional, Union

import numpy as np
import websockets
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid, Timeleft

from .async_concepts import (
    AsyncRecognitionBatch,
    AsyncRecognitionCallback,
    AsyncRecognizer,
    Recognition,
)
from .async_concepts import AsyncLoggerCallback
from .volcengine_bm_protocol import (
    VolcanoBigModelASRConfig,
    connect,
    send_init_request,
    send_audio,
    parse_response,
    Response,
    ResponseMessageType,
    FullServerResponse,
    nparray_to_bytes,
)


_ASR_CLOUD_BACKOFF_UNTIL = 0.0
_ASR_CLOUD_FAILURE_COUNT = 0
_ASR_CLOUD_LAST_ERROR = ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _asr_cloud_circuit_enabled() -> bool:
    return _env_bool("MOSS_ASR_CLOUD_CIRCUIT_BREAKER_ENABLED", True)


def _asr_force_local_only() -> bool:
    return _env_bool("MOSS_ASR_FORCE_LOCAL_ONLY", False)


def _asr_cloud_circuit_remaining() -> float:
    if not _asr_cloud_circuit_enabled():
        return 0.0
    return max(0.0, _ASR_CLOUD_BACKOFF_UNTIL - time.monotonic())


def _is_cloud_open_failure(error: Exception) -> bool:
    text = str(error).lower()
    return isinstance(error, (TimeoutError, asyncio.TimeoutError, OSError)) or (
        "opening handshake" in text or "connect" in text or "timed out" in text
    )


def _mark_asr_cloud_success() -> None:
    global _ASR_CLOUD_BACKOFF_UNTIL, _ASR_CLOUD_FAILURE_COUNT, _ASR_CLOUD_LAST_ERROR
    _ASR_CLOUD_BACKOFF_UNTIL = 0.0
    _ASR_CLOUD_FAILURE_COUNT = 0
    _ASR_CLOUD_LAST_ERROR = ""


def _mark_asr_cloud_failure(error: Exception, logger: LoggerItf | None = None) -> None:
    global _ASR_CLOUD_BACKOFF_UNTIL, _ASR_CLOUD_FAILURE_COUNT, _ASR_CLOUD_LAST_ERROR
    if not _asr_cloud_circuit_enabled() or not _is_cloud_open_failure(error):
        return
    _ASR_CLOUD_FAILURE_COUNT += 1
    _ASR_CLOUD_LAST_ERROR = str(error)
    threshold = max(1, _env_int("MOSS_ASR_CLOUD_CIRCUIT_FAILURE_THRESHOLD", 1))
    if _ASR_CLOUD_FAILURE_COUNT < threshold:
        return
    base_seconds = max(0.0, _env_float("MOSS_ASR_CLOUD_CIRCUIT_BACKOFF_SECONDS", 20.0))
    max_seconds = max(base_seconds, _env_float("MOSS_ASR_CLOUD_CIRCUIT_BACKOFF_MAX_SECONDS", 60.0))
    factor = max(1.0, _env_float("MOSS_ASR_CLOUD_CIRCUIT_BACKOFF_FACTOR", 1.5))
    seconds = min(max_seconds, base_seconds * (factor ** max(0, _ASR_CLOUD_FAILURE_COUNT - threshold)))
    _ASR_CLOUD_BACKOFF_UNTIL = max(_ASR_CLOUD_BACKOFF_UNTIL, time.monotonic() + seconds)
    if logger is not None:
        logger.warning(
            "ASR cloud circuit open for %.1fs after failure_count=%d error=%s",
            seconds,
            _ASR_CLOUD_FAILURE_COUNT,
            _ASR_CLOUD_LAST_ERROR[:160],
        )


def _reset_asr_cloud_circuit_for_tests() -> None:
    _mark_asr_cloud_success()


class AsyncLocalOnlyRecognitionBatch(AsyncRecognitionBatch):
    """ASR batch used while the cloud circuit is open.

    It deliberately performs no network I/O while still preserving all buffered
    audio for the listener state's local Whisper fallback.
    """

    def __init__(
        self,
        *,
        batch_id: str,
        logger: LoggerItf,
        reason: str = "",
    ):
        self.batch_id = batch_id or uuid()
        self.logger = logger
        self._reason = reason
        self._audio_buffer: deque[np.ndarray] = deque()
        self._started = False
        self._committed = False
        self._close_event = asyncio.Event()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.logger.warning(
            "Starting local-only ASR batch %s reason=%s",
            self.batch_id,
            self._reason[:160],
        )

    async def close(self, error: Optional[Exception] = None) -> None:
        if error is not None:
            self.logger.exception(error)
        self._close_event.set()
        self.logger.info(f"Local-only ASR batch {self.batch_id} closed")

    async def buffer(self, audio: np.ndarray) -> None:
        if self._close_event.is_set():
            self.logger.warning(f"Buffer closed for local-only batch {self.batch_id}")
            return
        self._audio_buffer.append(audio)

    async def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._close_event.set()
        self.logger.info(f"Committed local-only ASR batch {self.batch_id}")

    async def get_last_recognition(self) -> Optional[Recognition]:
        return None

    async def get_buffer(self) -> np.ndarray:
        if not self._audio_buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(list(self._audio_buffer))

    async def is_done(self) -> bool:
        return self._close_event.is_set()

    async def wait_until_done(self, timeout: Optional[float] = None) -> None:
        try:
            await asyncio.wait_for(self._close_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass


class AsyncVocEngineBigModelStreamASRBatch(AsyncRecognitionBatch):
    """
    异步火山引擎流式 ASR 批次。
    完全基于 asyncio，无线程。
    """

    def __init__(
        self,
        *,
        batch_id: str,
        config: VolcanoBigModelASRConfig,
        callback: AsyncRecognitionCallback,
        logger: LoggerItf,
        vad: Optional[int] = None,
        stop_on_sentence: bool = True,
    ):
        if not batch_id:
            batch_id = uuid()
        self.batch_id = batch_id
        self.config = config.resolve_env()
        self.logger = logger
        self.callback = callback
        self._started = False
        self._committed = False
        self._vad = vad
        self._stop_on_sentence = stop_on_sentence

        # 音频缓冲
        self._audio_buffer: deque[np.ndarray] = deque()
        self._audio_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(maxsize=100)
        """如果音频项为 None 表示要结束音频输出"""

        # 异步事件
        self._close_event = asyncio.Event()
        self._receiving_done = False
        self._sending_done = False

        # 识别结果
        self._last_recognition: Optional[Recognition] = None
        self._send_audio_seq = 0
        self._receive_rec_seq = 0

        # 主任务
        self._main_task: Optional[asyncio.Task] = None

    async def _main_loop(self) -> None:
        """主异步循环"""
        self.logger.info(f"Starting ASR batch {self.batch_id}")

        try:
            async with (await connect(self.config, self.batch_id)) as ws:
                _mark_asr_cloud_success()
                uid = self.batch_id
                await send_init_request(ws, self.config, uid, vad=self._vad)

                # 并发运行发送和接收任务
                sending_task = asyncio.create_task(self._send_audio_loop(ws))
                receiving_task = asyncio.create_task(self._receive_loop(ws))

                # 等待关闭事件或任务完成
                close_event_task = asyncio.create_task(self._close_event.wait())
                try:
                    await asyncio.wait(
                        [sending_task, receiving_task, close_event_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    # 取消未完成的任务
                    if not sending_task.done():
                        sending_task.cancel()
                    if not receiving_task.done():
                        receiving_task.cancel()
                    if not close_event_task.done():
                        close_event_task.cancel()

                    # 等待任务结束（带取消处理）
                    try:
                        await sending_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await receiving_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await close_event_task
                    except asyncio.CancelledError:
                        pass

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"Connection closed: {e}")
        except Exception as e:
            _mark_asr_cloud_failure(e, self.logger)
            self.logger.exception(e)
            await self.callback.on_error(f"ASR batch error: {e}")
        finally:
            self._close_event.set()
            # 防止异常情况无法发送尾包
            if self._last_recognition and not self._last_recognition.is_last:
                last_recognition = Recognition(
                    batch_id=self.batch_id,
                    text=self._last_recognition.text,
                    seq=self._last_recognition.seq,
                    sentence=self._last_recognition.sentence,
                    is_last=True,
                    created=time.time(),
                )
                await self.callback.on_recognition(last_recognition)

            # 保存音频
            await self._save_batch_audio()

            self.logger.info(f"ASR batch {self.batch_id} finished")

    async def _send_audio_loop(self, ws: websockets.ClientConnection) -> None:
        """发送音频数据循环"""
        try:
            while not self._close_event.is_set():
                try:
                    # 从队列获取音频数据，带超时
                    try:
                        audio_data = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        if self._committed:
                            # 已提交，发送None表示结束
                            audio_data = None
                        else:
                            continue

                    if audio_data is None:
                        # 发送尾包
                        self._sending_done = True
                        await self._send_audio_packet(ws, b"", sending_done=True)
                        self.logger.debug(f"Sent final packet for batch {self.batch_id}")
                        break

                    # 发送音频包。提交后也要先把队列里已采集的音频发完，
                    # 再由 None sentinel 发送尾包；否则短句容易被清空成空文本。
                    audio_bytes = nparray_to_bytes(audio_data)
                    self._send_audio_seq += 1
                    await self._send_audio_packet(ws, audio_bytes, sending_done=False)

                    self.logger.debug(
                        f"Sent audio packet seq {self._send_audio_seq} for batch {self.batch_id}"
                    )

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.exception(f"Error sending audio: {e}")
                    await self.callback.on_error(f"Send audio error: {e}")

        except asyncio.CancelledError:
            self.logger.info(f"Send audio loop cancelled for batch {self.batch_id}")
            raise
        except Exception as e:
            self.logger.exception(f"Send audio loop error: {e}")
            await self.callback.on_error(f"Send audio loop error: {e}")

    async def _send_audio_packet(
        self, ws: websockets.ClientConnection, audio_bytes: bytes, sending_done: bool
    ) -> None:
        """发送单个音频包"""
        try:
            await send_audio(ws, audio_bytes, self._send_audio_seq, sending_done)
        except websockets.exceptions.ConnectionClosed:
            raise
        except Exception as e:
            self.logger.exception(f"Error sending audio packet: {e}")
            raise

    async def _receive_loop(self, ws: websockets.ClientConnection) -> None:
        """接收识别结果循环"""
        try:
            while not self._close_event.is_set():
                try:
                    rec = await self._receive_single_response(ws)
                    if rec is None:
                        # 连接可能已关闭
                        continue

                    self._receive_rec_seq += 1

                    # 处理识别结果
                    if rec.sentence:
                        rec.is_last = self._stop_on_sentence or (self._committed and self._sending_done)

                    self._last_recognition = rec
                    await self.callback.on_recognition(rec)

                    if rec.is_last:
                        self.logger.info(f"Received final recognition for batch {self.batch_id}")
                        # 设置关闭事件，结束整个批次
                        self._close_event.set()
                        return

                except asyncio.CancelledError:
                    raise
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    self.logger.exception(f"Error receiving response: {e}")
                    await self.callback.on_error(f"Receive error: {e}")

        except asyncio.CancelledError:
            self.logger.info(f"Receive loop cancelled for batch {self.batch_id}")
            raise
        except Exception as e:
            self.logger.exception(f"Receive loop error: {e}")
            await self.callback.on_error(f"Receive loop error: {e}")
        finally:
            self._receiving_done = True

    async def _receive_single_response(
        self, ws: websockets.ClientConnection
    ) -> Optional[Recognition]:
        """接收单个响应"""
        try:
            data = await ws.recv()
            if not data:
                self.logger.error("Received empty data")
                return None

            response = parse_response(data)
            self.logger.debug(f"Parsed response: {response}")

            if response.message_type == ResponseMessageType.server_error:
                error = f"Server error: {response.error_code}, {response.payload}"
                await self.callback.on_error(error)
                self.logger.error(error)
                self._close_event.set()
                return None
            elif response.message_type == ResponseMessageType.server_ack:
                self.logger.debug(f"Server ACK: {response.payload}")
                return None
            elif response.message_type == ResponseMessageType.full_server_response:
                return self._handle_server_full_response(response)
            else:
                self.logger.info(f"Unknown message type: {response}")
                return None

        except websockets.exceptions.ConnectionClosed:
            raise
        except Exception as e:
            self.logger.exception(f"Error receiving single response: {e}")
            return None

    def _handle_server_full_response(self, response: Response) -> Recognition:
        """处理服务器完整响应"""
        data = json.loads(response.payload)
        fsp = FullServerResponse(**data)

        is_sentence = False
        if len(fsp.result.utterances) > 0:
            is_sentence = fsp.result.utterances[0].definite

        rec = Recognition(
            batch_id=self.batch_id,
            text=fsp.result.text,
            seq=response.sequence,
            sentence=is_sentence,
            is_last=is_sentence and self._committed,
            created=time.time(),
        )
        return rec

    async def _save_batch_audio(self) -> None:
        """保存批次音频"""
        if self._last_recognition:
            audio_buffer = await self.get_buffer()
            await self.callback.save_batch(self._last_recognition, audio_buffer)

    # AsyncRecognitionBatch 接口实现

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._main_task = asyncio.create_task(self._main_loop())
        self.logger.info(f"ASR batch {self.batch_id} started")

    async def close(self, error: Optional[Exception] = None) -> None:
        if error is not None:
            self.logger.exception(error)
            await self.callback.on_error(f"ASR batch closed with error: {error}")

        self._close_event.set()

        # 取消主任务
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"ASR batch {self.batch_id} closed")

    async def buffer(self, audio: np.ndarray) -> None:
        if self._close_event.is_set():
            self.logger.warning(f"Buffer closed for batch {self.batch_id}")
            return

        self._audio_buffer.append(audio)
        try:
            self._audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            # 队列满时丢弃最旧的数据，避免阻塞
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(audio)
            except asyncio.QueueEmpty:
                pass

    async def commit(self) -> None:
        if self._committed:
            return

        self._committed = True
        self.logger.info(f"Committed ASR batch {self.batch_id}")

        # 通知发送循环可以结束：保留队列中已采集的音频，最后追加 sentinel。
        try:
            self._audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(None)
            except asyncio.QueueEmpty:
                pass

    async def get_last_recognition(self) -> Optional[Recognition]:
        return self._last_recognition

    async def get_buffer(self) -> np.ndarray:
        if not self._audio_buffer:
            return np.array([], dtype=np.int16)

        # 合并所有音频数据
        combined = np.concatenate(list(self._audio_buffer))
        return combined

    async def is_done(self) -> bool:
        return self._close_event.is_set()

    async def wait_until_done(self, timeout: Optional[float] = None) -> None:
        try:
            await asyncio.wait_for(self._close_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass


class AsyncVocEngineBigModelASR(AsyncRecognizer):
    """异步火山引擎语音识别引擎"""

    def __init__(
        self,
        *,
        config: VolcanoBigModelASRConfig,
        logger: LoggerItf,
        callback: AsyncRecognitionCallback = None,
    ):
        self.config = config.resolve_env()
        self.logger = logger
        self.sample_rate = config.sample_rate
        self.frame_duration = config.frame_time / 1000
        self._closed = False
        self.callback = callback or AsyncLoggerCallback(self.logger)

    async def start(self) -> None:
        """启动识别引擎（异步无操作）"""
        pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    async def is_closed(self) -> bool:
        return self._closed

    async def new_batch(
        self,
        callback: AsyncRecognitionCallback = None,
        batch_id: str = "",
        vad: Optional[int] = None,
        stop_on_sentence: bool = False,
    ) -> AsyncRecognitionBatch:
        if callback is None:
            callback = AsyncLoggerCallback(self.logger)

        if _asr_force_local_only():
            self.logger.warning("ASR force-local-only enabled; creating local-only batch")
            return AsyncLocalOnlyRecognitionBatch(
                batch_id=batch_id,
                logger=self.logger,
                reason="force_local_only",
            )

        cloud_remaining = _asr_cloud_circuit_remaining()
        if cloud_remaining > 0.0:
            self.logger.warning(
                "ASR cloud circuit still open for %.1fs; creating local-only batch reason=%s",
                cloud_remaining,
                _ASR_CLOUD_LAST_ERROR[:160],
            )
            return AsyncLocalOnlyRecognitionBatch(
                batch_id=batch_id,
                logger=self.logger,
                reason=_ASR_CLOUD_LAST_ERROR,
            )

        return AsyncVocEngineBigModelStreamASRBatch(
            callback=callback,
            batch_id=batch_id,
            config=self.config,
            logger=self.logger,
            vad=vad,
            stop_on_sentence=stop_on_sentence,
        )
