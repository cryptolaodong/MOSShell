import enum
import gzip
import io
import json
import os
import struct
import uuid
from typing import Optional, List, NamedTuple

import numpy as np
import websockets
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    'VolcanoBigModelASRConfig',
    'connect',
    'create_init_request', 'send_init_request', 'send_audio', 'create_audio_only_request',
    'parse_response',
    'Response', 'ResponseMessageType',
    'FullServerResponse', 'Result', 'Utterance', 'Word', 'AudioInfo',
    'nparray_to_bytes',
]


class VolcanoBigModelASRConfig(BaseModel):
    """
    火山引擎 asr 配置项.
    """
    appid: str = Field("$VOLCENGINE_BM_ASR_APPID", description="火山引擎 asr 的 appid")
    token: str = Field("$VOLCENGINE_BM_ASR_TOKEN", description="火山引擎的 asr app token")
    url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    sample_rate: int = Field(16000, description="默认的采样率")
    bits: int = Field(16)
    channel: int = Field(1)
    model_name: str = Field("bigmodel", description="火山引擎模型类型")
    end_window_size: int = Field(500, description="单位ms，默认为800，最小200。"
                                                  "静音时长超过该值，会直接判停，输出definite。"
                                                  "配置该值，不使用语义分句，根据静音时长来分句。"
                                                  "用于实时性要求较高场景，可以提前获得definite句子")
    frame_time: int = Field(128, description="预期每一帧的时间长度. 单位是 ms")
    enable_punc: bool = Field(True, description="启用标点")
    enable_ddc: bool = Field(True, description="**语义顺滑**‌是一种技术，旨在提高自动语音识别（ASR）结果的文本可读性和流畅性。"
                                               "这项技术通过删除或修改ASR结果中的不流畅部分，如停顿词、语气词、语义重复词等，"
                                               "使得文本更加易于阅读和理解。")
    resource_id: str = Field("volc.bigasr.sauc.duration")

    def resolve_env(self) -> Self:
        """
        如果appid/token以$开头，则从环境变量读取
        """
        if self.appid.startswith("$"):
            appid = self.appid
            self.appid = os.environ.get(appid[1:], appid)
        if self.token.startswith("$"):
            token = self.token
            self.token = os.environ.get(token[1:], token)
        return self


# ----------------------------------------主ASR实现---------------------------------

# 内部协议处理类
class _Protocol:
    """内部协议处理类"""

    # 协议常量
    PROTOCOL_VERSION = 0x01
    DEFAULT_HEADER_SIZE = 0x01

    # 消息类型
    FULL_CLIENT_REQUEST = 0x01
    AUDIO_ONLY_REQUEST = 0x02
    FULL_SERVER_RESPONSE = 0x09
    SERVER_ACK = 0x0B
    SERVER_ERROR_RESPONSE = 0x0F

    # 消息类型特定标志
    NO_SEQUENCE = 0x00
    POS_SEQUENCE = 0x01
    NEG_SEQUENCE = 0x02
    NEG_WITH_SEQUENCE = 0x03

    # 序列化方法
    NO_SERIALIZATION = 0x00
    JSON = 0x01

    # 压缩方式
    NO_COMPRESSION = 0x00
    GZIP = 0x01

    @staticmethod
    def get_header(message_type: int, message_type_specific_flags: int,
                   serial_method: int, compression_type: int, reserved_data: int = 0) -> bytes:
        """生成协议头部"""
        header = bytearray(4)
        header[0] = (_Protocol.PROTOCOL_VERSION << 4) | _Protocol.DEFAULT_HEADER_SIZE
        header[1] = (message_type << 4) | message_type_specific_flags
        header[2] = (serial_method << 4) | compression_type
        header[3] = reserved_data
        return bytes(header)

    @staticmethod
    def int_to_bytes(value: int) -> bytes:
        """整数转换为4字节"""
        # 用负数序列号，应该用有符号整型，即'>i'而不是'>I'
        return struct.pack('>i', value)

    @staticmethod
    def gzip_compress(data: bytes) -> bytes:
        """GZIP压缩"""
        if not data:
            return b''

        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb') as f:
            f.write(data)
        return buf.getvalue()

    @staticmethod
    def gzip_decompress(data: bytes) -> bytes:
        """GZIP解压"""
        if not data:
            return b''

        buf = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buf, mode='rb') as f:
            return f.read()


async def connect(config: VolcanoBigModelASRConfig, connection_id: str = "") -> websockets.ClientConnection:
    config = config.resolve_env()
    connection_id = connection_id or uuid()
    try:
        open_timeout = float(os.environ.get("MOSS_ASR_WS_OPEN_TIMEOUT_SECONDS", "4.0"))
    except Exception:
        open_timeout = 4.0
    open_timeout_arg = open_timeout if open_timeout > 0 else None
    headers = {
        "X-Api-App-Key": config.appid,
        "X-Api-Access-Key": config.token,
        "X-Api-Resource-Id": config.resource_id,
        "X-Api-Connect-Id": connection_id,
    }
    ws = await websockets.connect(
        config.url,
        additional_headers=headers,
        open_timeout=open_timeout_arg,
    )
    return ws


def create_init_request(uid: str, config: VolcanoBigModelASRConfig, vad: Optional[int] = None) -> tuple[bytes, int]:
    """创建初始化请求"""
    # 构建请求JSON
    # todo: 如果开发时间充分, 它应该是一个 base model.
    payload = {
        "user": {"uid": uid},
        "audio": {
            "format": "pcm",
            "sample_rate": config.sample_rate,
            "bits": config.bits,
            "channel": config.channel,
            "codec": "raw"
        },
        "request": {
            "model_name": config.model_name,
            "enable_punc": config.enable_punc,
            "end_window_size": vad or config.end_window_size,
            "force_to_speech_time": 1000,
            "show_utterances": True,
        }
    }

    payload_str = json.dumps(payload)

    # 压缩payload
    payload_bytes = _Protocol.gzip_compress(payload_str.encode('utf-8'))

    # 设置序列号
    seq = 1
    seq_bytes = _Protocol.int_to_bytes(seq)

    # 构建header
    header = _Protocol.get_header(
        _Protocol.FULL_CLIENT_REQUEST,
        _Protocol.POS_SEQUENCE,
        _Protocol.JSON,
        _Protocol.GZIP
    )

    # 构建payload长度
    payload_size = _Protocol.int_to_bytes(len(payload_bytes))

    # 组装完整消息
    message = header + seq_bytes + payload_size + payload_bytes
    return message, seq


def nparray_to_bytes(audio: np.ndarray) -> bytes:
    # 将numpy数组转换为字节
    audio_bytes = audio.tobytes()
    # 压缩音频数据
    compressed_audio = _Protocol.gzip_compress(audio_bytes)
    return compressed_audio


def create_audio_only_request(audio: bytes, seq: int, is_last: bool = False) -> tuple[bytes, int]:
    """创建音频数据消息"""
    # 增加序列号
    seq += 1
    seq_value = -seq if is_last else seq
    seq_bytes = _Protocol.int_to_bytes(seq_value)

    # 计算压缩后的大小
    payload_size = _Protocol.int_to_bytes(len(audio))

    # 构建header
    message_flags = _Protocol.NEG_WITH_SEQUENCE if is_last else _Protocol.POS_SEQUENCE
    header = _Protocol.get_header(
        _Protocol.AUDIO_ONLY_REQUEST,
        message_flags,
        _Protocol.JSON,
        _Protocol.GZIP
    )

    # 组装完整消息
    message = b''.join([header, seq_bytes, payload_size, audio])
    return message, seq


async def send_init_request(ws: websockets.ClientConnection, config: VolcanoBigModelASRConfig, uid: str,
                            vad: Optional[int] = None) -> None:
    """发送初始化请求"""
    message, seq = create_init_request(uid, config, vad)
    await ws.send(message)


async def send_audio(ws: websockets.ClientConnection, audio: bytes, seq: int, is_last: bool = False) -> None:
    """发送音频数据"""
    message, seq = create_audio_only_request(audio, seq, is_last)
    await ws.send(message)


_Sequence = int


class ResponseMessageType(str, enum.Enum):
    full_server_response = "full_server_response"
    server_error = "server_error"
    server_ack = "server_ack"


class Response(NamedTuple):
    sequence: int
    message_type: Optional[ResponseMessageType]
    error_code: Optional[int]
    is_last: bool
    payload: str


def parse_response(data: bytes) -> Response:
    """
    解析服务器响应
    """
    # 解析协议头
    message_type = (data[1] >> 4) & 0x0F
    message_type_specific_flags = data[1] & 0x0F
    message_compression = data[2] & 0x0F

    # 解析序列号
    sequence = struct.unpack('>I', data[4:8])[0]

    # 解析payload大小
    payload_size = struct.unpack('>I', data[8:12])[0]

    # 提取payload
    payload = data[12:12 + payload_size] if len(data) >= 12 + payload_size else data[12:]
    # payload = data[header_size * 4:]

    is_last_package = False
    if message_type_specific_flags & 0x02:
        # receive last package
        is_last_package = True

    if message_type == _Protocol.FULL_SERVER_RESPONSE:
        # 完整服务器响应
        if message_compression == _Protocol.GZIP:
            decompressed = _Protocol.gzip_decompress(payload)
            payload_str = decompressed.decode('utf-8')
        else:
            payload_str = payload.decode('utf-8')
        return Response(
            sequence=sequence,
            message_type=ResponseMessageType.full_server_response,
            error_code=None,
            is_last=is_last_package,
            payload=payload_str,
        )

    elif message_type == _Protocol.SERVER_ACK:
        return Response(
            sequence=sequence,
            message_type=ResponseMessageType.server_ack,
            error_code=None,
            is_last=False,
            payload="",
        )

    elif message_type == _Protocol.SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        payload_msg = payload[8:]
        if message_compression == _Protocol.GZIP:
            payload_msg = gzip.decompress(payload_msg)

        # 错误响应
        return Response(
            sequence=sequence,
            message_type=ResponseMessageType.server_error,
            error_code=code,
            is_last=is_last_package,
            payload=payload_msg,
        )
    else:
        return Response(
            sequence=-1,
            message_type=ResponseMessageType.server_error,
            error_code=-1,
            is_last=False,
            payload="unknown error",
        )


class AudioInfo(BaseModel):
    duration: int = Field(default=0, description="单位是毫秒")


class Word(BaseModel):
    blank_duration: int = Field(0)
    end_time: int = Field(0)
    start_time: int = Field(0)
    text: str = Field("")


class Utterance(BaseModel):
    definite: bool = Field(default=False, description="是否是一个确定的分句")
    end_time: int = Field(default=0, description="结束时间")
    start_time: int = Field(default=0, description="开始时间")
    text: str = Field(default="", description="分句")
    words: List[Word] = Field(default_factory=list)


class Result(BaseModel):
    text: str = Field(description="识别结果")
    utterances: List[Utterance] = Field(default_factory=list)


class FullServerResponse(BaseModel):
    audio_info: AudioInfo = Field(default_factory=AudioInfo)
    result: Result = Field(default_factory=Result)
