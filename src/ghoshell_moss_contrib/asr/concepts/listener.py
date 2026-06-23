import time
from abc import abstractmethod
from enum import Enum
from threading import Event, Thread
from typing import (
    Optional, List, Union, Protocol, Callable,
    NamedTuple,
)

import numpy as np
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field

__all__ = [
    'ListenerService',
    'ListenerCallback', 'ListenerState', 'ListenerStateName',
    "AudioInput", 'AudioInputLoop',
    'Recognition', 'RecognitionCallback', 'RecognitionBatch', "Recognizer",
]


class AudioInput(Protocol):
    """
    音频输入模块.
    从音频输入 (比如 mic) 中获取一段音频.
    可以通过 adapter 嵌套来解决转码等问题.
    """
    input_id: str
    rate: int
    channels: int
    dtype: np.dtype

    @abstractmethod
    def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        """获取一段音频. 理论上需要连续获取音频."""
        pass

    @abstractmethod
    def start(self) -> None:
        """启动音频输入. 理论上可以启动 n 次. 但 close 后就不能启动了. """
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止音频但不回收资源. 必须要 start 才能重启. """
        pass

    @abstractmethod
    def close(self, error: Optional[Exception] = None) -> None:
        """关闭音频输入, 回收资源."""
        pass

    @abstractmethod
    def closed(self) -> bool:
        pass

    def __enter__(self) -> "AudioInput":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exc_val)


class AudioInputLoop:
    """
    一个基线示例, 如何获得连续的音频.
    """

    def __init__(
            self,
            send: Callable[[np.ndarray], None],
            audio_input: AudioInput,
            *,
            stop_event: Optional[Event] = None,
            resample_rate: Optional[int] = None,
            frame: Optional[float] = None,
    ):
        self._audio_input = audio_input
        self._resample_rate = resample_rate
        self._frame = frame
        self._output_fn = send
        self._stop = stop_event or Event()
        self._main_loop_done = Event()
        self._exception = None
        self._main_loop_thread = Thread(target=self._main_loop, daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._main_loop_thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> bool:
        return self._main_loop_done.wait(timeout)

    def _main_loop(self) -> None:
        audio_input = self._audio_input
        try:
            self._audio_input.start()
            while not self._stop.is_set() and not audio_input.closed():
                data = audio_input.read(rate=self._resample_rate, duration=self._frame)
                self._output_fn(data)
        except Exception as e:
            self._exception = e
        finally:
            audio_input.stop()
            self._main_loop_done.set()
            self._stop.set()
            self._audio_input.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._exception:
            raise self._exception


class Recognition(BaseModel):
    """
    音频解析结果.
    一个可扩展的数据结构.
    未来可能会有开始结束时间等.
    """
    batch_id: str = Field(default_factory=uuid, description="批次id, 也对应音频文件")
    seq: int = Field(default=0)
    text: str = Field(default="", description="识别文本")
    sentence: bool = Field(default=False, description="是否是一个完整的分句. ")

    is_last: bool = Field(False, description="是不是批次里最后一条识别数据")
    created: float = Field(default_factory=lambda: round(time.time(), 4), description="创建时间")
    commit_reason: str = Field(default="", description="提交原因，如 energy_vad / empty_text_timeout / manual")
    audio_max_rms: float = Field(default=0.0, description="本批次识别前观测到的最大音频 RMS")


class RecognitionCallback(Protocol):
    """
    音频事件回调接口.
    用于处理各种音频事件, 如唤醒、VAD、识别结果等.
    实现者可以选择只实现关心的方法.
    """

    @abstractmethod
    def on_recognition(self, result: Recognition) -> None:
        """
        收到识别结果时的回调.
        包括中间结果和最终结果.
        :param result: 识别结果
        """
        pass

    def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        """
        完成时回调保存 batch
        """
        # 默认先不实现.
        return

    @abstractmethod
    def on_error(self, error: str) -> None:
        """
        接受到异常.
        """
        pass


class RecognitionBatch(Protocol):
    """
    单个语音识别会话.
    代表一次完整的识别流程 (从开始说话到结束).
    可以持续追加音频数据, 并获得实时识别结果.
    """
    batch_id: str

    @abstractmethod
    def start(self) -> None:
        """
        启动会话.
        """
        pass

    @abstractmethod
    def close(self, error: Optional[Exception] = None) -> None:
        """
        关闭会话, 释放资源.
        """
        pass

    @abstractmethod
    def buffer(self, audio: np.ndarray) -> None:
        """
        异步追加音频数据.

        :param audio: 音频数据
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        含义是拿到下一个分句就退出.
        """
        pass

    @abstractmethod
    def get_last_recognition(self) -> Union[Recognition, None]:
        pass

    @abstractmethod
    def get_buffer(self) -> np.ndarray:
        """
        获取当前会话的完整音频缓冲.
        :returns: 完整的音频数据
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        会话是否已完成.
        :returns: 会话是否完成
        """
        pass

    @abstractmethod
    def wait_until_done(self, timeout: Optional[float] = None) -> None:
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.close(exc_val)
        else:
            self.commit()
            self.wait_until_done()


class Recognizer(Protocol):
    """
    语音识别引擎.
    管理多个识别会话, 提供统一的识别接口.
    """
    frame_duration: float
    """每一帧的采样时间. 单位是秒"""
    sample_rate: int
    """每一帧的期望 rate"""

    @abstractmethod
    def start(self) -> None:
        """
        启动识别引擎.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭识别引擎.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        判断引擎是否已关闭.

        :returns: 是否已关闭
        """
        pass

    @abstractmethod
    def new_batch(
            self,
            callback: RecognitionCallback = None,
            batch_id: str = "",
            vad: Optional[float] = None,
            stop_on_sentence: bool = False,
    ) -> RecognitionBatch:
        """
        创建新的识别会话. 需要根据输入音频的 sample_rate
        :param callback:
        :param batch_id:
        :param vad: 单位是毫秒.
        :param stop_on_sentence: 如果为 True, 在拿到分句时就会结束 batch. 否则必须等待 commit
        """
        pass


class ListenerCallback(RecognitionCallback, Protocol):

    @abstractmethod
    def on_waken(self) -> None:
        pass

    @abstractmethod
    def on_state_change(self, state: str) -> None:
        pass


class ListenerStateName(str, Enum):
    listening = "listening"
    deaf = "deaf"
    asleep = "asleep"
    pdt_listening = "pdt_listening"
    pdt_waiting = "pdt_waiting"


class ListenerState(Protocol):
    class NextState(NamedTuple):
        state_name: str
        audio_buffer: Optional[np.ndarray]

    @abstractmethod
    def name(self) -> ListenerStateName:
        pass

    @abstractmethod
    def clear_buffer(self) -> None:
        """
        清空所有缓冲.
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        主动提交结束状态. 开始等待状态变更.
        """
        pass

    @abstractmethod
    def set_vad(self, vad_time: int) -> None:
        pass

    @abstractmethod
    def next(self) -> Union[NextState, None]:
        """
        如果 NextState 不为空, 当前的状态机会被切换状态.
        :return: next state
        """
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class ListenerService(Protocol):
    """
    音频输入服务.
    集成音频采集、事件检测和语音识别功能,
    提供完整的语音交互服务.

    支持多种部署模式:
    1. 线程模式: 将它在一个线程内独立运行
    2. 多进程模式: 独立进程, 通过 Queue 通讯
    3. HTTP/RPC 模式: 独立服务, 接受 API 调用
    4. 事件驱动: 如通过 ROS 事件来调用自身 API
    """

    @abstractmethod
    def bootstrap(self):
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def audio_input(self) -> AudioInput:
        pass

    @abstractmethod
    def recognizer(self) -> Recognizer:
        pass

    @abstractmethod
    def set_callback(self, callback: ListenerCallback) -> None:
        """
        设置回调.
        """
        pass

    @abstractmethod
    def clear_buffer(self) -> None:
        """
        清空所有缓冲.
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        主动提交会话, 获取最终结果.
        """
        pass

    @abstractmethod
    def set_vad(self, vad_time: int) -> None:
        pass

    @abstractmethod
    def all_states(self) -> List[str]:
        pass

    @abstractmethod
    def set_state(self, state: str) -> None:
        pass

    @abstractmethod
    def current_state(self) -> ListenerState:
        pass
