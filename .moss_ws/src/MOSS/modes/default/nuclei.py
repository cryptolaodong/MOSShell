# 感知核声明 — Mindflow 输入信号源的声明。
from ghoshell_moss.core.mindflow.input_signal_nucleus import InputSignalNucleus
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta, Nucleus, Priority, InputSignal
from ghoshell_container import IoCContainer


class InputNucleusMeta(NucleusMeta):
    """用户输入信号核 — 监听 input signal, 驱动 AI 思考。"""

    def name(self) -> str:
        return "input_nucleus"

    def description(self) -> str:
        return "User input signal handler — text and voice input"

    def signals(self) -> list:
        return [InputSignal]

    def factory(self, container: IoCContainer) -> Nucleus:
        return InputSignalNucleus(
            name="input_nucleus",
            description="Aggregates user text/voice input signals",
            target_signal="input",
            suppress_seconds=0.3,
            buffer_size=10,
            min_priority=Priority.INFO,
        )


input_nucleus_meta = InputNucleusMeta()


class VoiceNucleusMeta(NucleusMeta):
    """语音感知核 — 持续监听机器人麦克风。"""

    def name(self) -> str:
        return "voice_nucleus"

    def description(self) -> str:
        return "Robot microphone listener — continuous voice detection"

    def signals(self) -> list:
        return []

    def factory(self, container: IoCContainer) -> Nucleus:
        from ghoshell_moss.contracts.logger import LoggerItf
        from .voice_nucleus import VoiceNucleus
        logger = container.get(LoggerItf)
        return VoiceNucleus(container=container, logger=logger)


voice_nucleus_meta = VoiceNucleusMeta()
