"""
异步 Listener 服务实现。

完全基于 asyncio，弃用线程模型。
"""

import asyncio
from typing import Optional, List
import numpy as np
# pyaudio imported lazily below
from ghoshell_common.contracts import LoggerItf

from .async_concepts import (
    AsyncListenerService,
    AsyncListenerState,
    AsyncListenerCallback,
    AsyncAudioInput,
    AsyncRecognizer,
    AsyncListenerStateName,
)
from .async_states import (
    AsyncDeafState,
    AsyncListeningState,
    AsyncPdtListeningState,
    AsyncPdtWaitingState,
)
# AsyncPyAudioInput imported lazily
from .async_volcengine_bm import AsyncVocEngineBigModelASR
from .configs import ListenerConfig
from .async_concepts import AsyncLoggerCallback


class AsyncListenerServiceImpl(AsyncListenerService):
    """
    异步 Listener 服务实现。
    """

    def __init__(
        self,
        config: ListenerConfig,
        *,
        logger: LoggerItf,
        default_state_name: str = "",
        callback: Optional[AsyncListenerCallback] = None,
        audio_input: Optional[AsyncAudioInput] = None,
    ):
        self._config = config.resolve_env()
        self._logger = logger
        self._callback = callback or AsyncLoggerCallback(logger)
        self._default_state_name = default_state_name or config.default_state_name

        # 初始化音频输入
        if audio_input is None:
            # 仅在没有外部音频输入时创建 PyAudio 实例
            import pyaudio
            self._pa = pyaudio.PyAudio()
            audio_input_config = self._config.get_audio_input_config()
            # 注意：这里需要异步创建，但在构造函数中无法异步
            # 我们将在 bootstrap 中完成
            self._audio_input_config = audio_input_config
            self._audio_input: Optional[AsyncAudioInput] = None
        else:
            self._pa = None
            self._audio_input = audio_input
            self._audio_input_config = None

        # 初始化识别器
        self._recognizer = self._make_recognizer()

        # 状态管理
        self._current_state: Optional[AsyncListenerState] = None
        self._next_state_request: Optional[tuple[str, Optional[np.ndarray]]] = None
        self._bootstrapped = False
        self._closed = False
        self._state_loop_task: Optional[asyncio.Task] = None
        self._state_change_lock = asyncio.Lock()

    def _make_recognizer(self) -> AsyncRecognizer:
        """创建异步识别器"""
        if self._config.use_asr == "volcengine_bm_asr":
            return AsyncVocEngineBigModelASR(
                config=self._config.volcengine_bm_asr,
                logger=self._logger,
                callback=self._callback,
            )
        else:
            raise NotImplementedError(f"{self._config.use_asr} is not supported")

    async def _make_state(
        self, state_name: str, buffer: Optional[np.ndarray]
    ) -> AsyncListenerState:
        """创建异步状态实例"""
        if state_name == AsyncListenerStateName.LISTENING.value:
            return AsyncListeningState(
                recognizer=self._recognizer,
                audio_input=await self.audio_input(),
                callback=self._callback,
                logger=self._logger,
                vad=None,  # 本地 VAD 未实现
                stop_on_sentence=True,
                on_complete_state=self._config.asr_on_vad_state or "",
                max_idle_time=self._config.asr_max_idle_time,
                on_max_idle_state=self._config.asr_on_idle_state or "",
                allow_batch=0,
            )
        elif state_name == AsyncListenerStateName.PDT_LISTENING.value:
            from .vad import EnergyVAD
            return AsyncPdtListeningState(
                recognizer=self._recognizer,
                audio_input=await self.audio_input(),
                callback=self._callback,
                logger=self._logger,
                vad=EnergyVAD(),
            )
        elif state_name == AsyncListenerStateName.PDT_WAITING.value:
            return AsyncPdtWaitingState()
        elif state_name == AsyncListenerStateName.ASLEEP.value:
            # 唤醒词功能跳过，退化为聋状态
            self._logger.info("Asleep state requested but wake word功能跳过，使用DeafState")
            return AsyncDeafState(logger=self._logger)
        else:
            # 默认返回聋状态
            return AsyncDeafState(logger=self._logger)

    async def audio_input(self) -> AsyncAudioInput:
        """获取音频输入（延迟初始化）"""
        if self._audio_input is None and self._audio_input_config is not None:
            # 异步创建音频输入
            from .async_pyaudio_input import AsyncPyAudioInputConfig
            async_config = AsyncPyAudioInputConfig(self._audio_input_config)
            self._audio_input = await async_config.new_audio_input(
                pa=self._pa,
                logger=self._logger,
                dtype=np.int16,
            )
        if self._audio_input is None:
            raise RuntimeError("Audio input not initialized")
        return self._audio_input

    async def recognizer(self) -> AsyncRecognizer:
        return self._recognizer

    async def set_callback(self, callback: AsyncListenerCallback) -> None:
        self._callback = callback

    async def clear_buffer(self) -> None:
        if self._current_state:
            await self._current_state.clear_buffer()

    async def commit(self) -> None:
        if self._current_state:
            await self._current_state.commit()

    async def set_vad(self, vad_time: int) -> None:
        if self._current_state:
            await self._current_state.set_vad(vad_time)

    async def all_states(self) -> List[str]:
        # 返回所有支持的状态
        return [
            AsyncListenerStateName.LISTENING.value,
            AsyncListenerStateName.DEAF.value,
            AsyncListenerStateName.ASLEEP.value,
            AsyncListenerStateName.PDT_LISTENING.value,
            AsyncListenerStateName.PDT_WAITING.value,
        ]

    async def set_state(self, state: str) -> None:
        """设置状态（强制切换）"""
        if state not in await self.all_states():
            raise ValueError(f"State {state} not supported")

        self._next_state_request = (state, None)
        self._logger.info(f"State change requested to {state}, current state: {self._current_state.name() if self._current_state else 'None'}")

    async def current_state(self) -> AsyncListenerState:
        if self._current_state is None:
            raise RuntimeError("No current state")
        return self._current_state

    async def bootstrap(self) -> None:
        """启动服务"""
        if self._bootstrapped:
            return

        self._bootstrapped = True
        self._closed = False

        # 确保音频输入已初始化
        await self.audio_input()

        # 创建初始状态
        initial_state = await self._make_state(self._default_state_name, None)
        await initial_state.start()
        self._current_state = initial_state

        # 启动状态循环
        self._state_loop_task = asyncio.create_task(self._state_loop())

        self._logger.info(f"AsyncListenerService bootstrapped with state {self._default_state_name}")
        self._logger.info(f"State loop task started: {self._state_loop_task}")

    async def shutdown(self) -> None:
        """关闭服务"""
        if self._closed:
            return

        self._closed = True
        self._bootstrapped = False

        # 取消状态循环
        if self._state_loop_task and not self._state_loop_task.done():
            self._state_loop_task.cancel()
            try:
                await self._state_loop_task
            except asyncio.CancelledError:
                pass

        # 关闭当前状态
        if self._current_state:
            await self._current_state.close()

        # 关闭识别器
        await self._recognizer.close()

        # 关闭音频输入
        if self._audio_input:
            await self._audio_input.close()

        # 关闭 PyAudio（仅当自己创建了实例时）
        if self._pa is not None:
            self._pa.terminate()

        self._logger.info("AsyncListenerService shutdown complete")

    async def _state_loop(self) -> None:
        """状态循环"""
        self._logger.info("State loop started")
        while not self._closed:
            try:
                # 检查状态切换请求
                if await self._check_state_change():
                    self._logger.debug("State change processed, continuing loop")
                    continue

                # 检查当前状态是否建议切换
                if self._current_state:
                    next_state_info = await self._current_state.next_state()
                    if next_state_info:
                        state_name, buffer = next_state_info
                        self._logger.info(f"Current state suggests switching to {state_name}")
                        self._next_state_request = (state_name, buffer)

                # 短暂休眠
                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                self._logger.info("State loop cancelled")
                break
            except Exception as e:
                self._logger.exception(f"Error in state loop: {e}")
                await self._callback.on_error(f"State loop error: {e}")
                await asyncio.sleep(1.0)  # 避免频繁错误循环

    async def _check_state_change(self) -> bool:
        """检查并执行状态切换"""
        if self._next_state_request is None:
            return False

        async with self._state_change_lock:
            if self._next_state_request is None:
                return False

            state_name, buffer = self._next_state_request
            self._next_state_request = None

            try:
                self._logger.info(f"Changing state to {state_name}")

                # 关闭当前状态
                if self._current_state:
                    await self._current_state.close()

                # 创建新状态
                new_state = await self._make_state(state_name, buffer)
                await new_state.start()
                self._current_state = new_state

                # 通知回调
                await self._callback.on_state_change(state_name)

                self._logger.info(f"State changed to {state_name}")
                return True

            except Exception as e:
                self._logger.exception(f"Error changing state to {state_name}: {e}")
                await self._callback.on_error(f"State change error: {e}")

                # 尝试恢复到安全状态（聋状态）
                try:
                    safe_state = AsyncDeafState(logger=self._logger)
                    await safe_state.start()
                    self._current_state = safe_state
                    await self._callback.on_state_change(AsyncListenerStateName.DEAF.value)
                except Exception as recovery_error:
                    self._logger.exception(f"Error recovering to safe state: {recovery_error}")

                return False

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()