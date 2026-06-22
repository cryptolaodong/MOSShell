import logging

from ghoshell_container import IoCContainer
from reachy_mini import ReachyMini

from ghoshell_moss import new_prime_channel, Message, Text, Channel, Matrix
from ghoshell_moss.core.blueprint.channel_builder import ChannelCreator
from ghoshell_moss.contracts import Workspace, LoggerItf
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.py_channel import StatefulChannelRuntimeImpl
from ghoshell_moss.core.concepts.command import PyCommand
from ghoshell_moss_contrib.moss_in_reachy_mini.components.antennas import Antennas
from ghoshell_moss_contrib.moss_in_reachy_mini.components.body import Body
from ghoshell_moss_contrib.moss_in_reachy_mini.components.head import Head
from ghoshell_moss_contrib.moss_in_reachy_mini.components.vision import Vision
from ghoshell_moss_contrib.moss_in_reachy_mini.state.asleep import AsleepState
from ghoshell_moss_contrib.moss_in_reachy_mini.state.boring import BoringState
from ghoshell_moss_contrib.moss_in_reachy_mini.state.waken import WakenState

__all__ = ['MossInReachyMini', 'ReachyMiniChannelCreator', 'provide_channel']


class MossInReachyMini:
    """Reachy Mini 躯体控制器 — 组装 StatefulChannel，状态切换由 MOSS runtime 管理。"""

    def __init__(
            self,
            mini: ReachyMini,
            ws: Workspace,
            logger: LoggerItf,
            default_state: str = "waken",
            name: str = "reachy_mini",
            description: str = "reachy mini robot",
            allow_vision: bool = False,
    ):
        self._mini = mini
        self.logger = logger
        self._default_state = default_state
        self._name = name
        self._description = description
        self._allow_vision = allow_vision

        # layer 1: body components
        self._head = Head(mini)
        self._body = Body(mini, ws, logger)
        self._antennas = Antennas(mini, logger=logger)

        # layer 2: media channels
        self._vision = Vision(mini, logger=logger)

        # layer 3: states
        self._waken = WakenState(mini, self._body, self._head, self._antennas)
        self._boring = BoringState(mini)
        self._asleep = AsleepState(mini)

    # -- context messages -----------------------------------------------------

    async def context_messages(self):
        messages = []

        state_message = Message.new(name="__reachy_mini_state__").with_content(
            Text(text="Reachy Mini body channel — 各状态下可用命令通过 switch_state 切换后暴露"),
        )
        messages.append(state_message)

        return messages

    # -- channel assembly -----------------------------------------------------

    def as_channel(self) -> Channel:
        self.logger.info("MossInReachyMini.as_channel()")
        states = {
            self._waken.name(): self._waken,
            self._boring.name(): self._boring,
            self._asleep.name(): self._asleep,
        }

        channel = new_prime_channel(
            name=self._name or "reachy_mini",
            description=self._description or "reachy mini root channel",
        )

        channel.build.context_messages(self.context_messages)
        channel.build.startup(self.bootstrap)
        channel.build.close(self.aclose)

        # Register core body commands on main_state so they're always available,
        # even if _current_state is None during clear/refresh cycles.
        channel.build.add_command(PyCommand(
            self._body.dance, name="dance",
            doc=self._body.dance_docstring, blocking=True,
        ))
        channel.build.add_command(PyCommand(
            self._body.emotion, name="emotion",
            doc=self._body.emotion_docstring, blocking=True,
        ))
        channel.build.add_command(PyCommand(
            self._head.move, name="head_move", blocking=True,
        ))
        channel.build.add_command(PyCommand(
            self._head.reset, name="head_reset", blocking=True,
        ))
        channel.build.add_command(PyCommand(
            self._antennas.move, name="antennas_move", blocking=True,
        ))
        channel.build.add_command(PyCommand(
            self._antennas.reset, name="antennas_reset", blocking=True,
        ))

        default_state = self._default_state
        if not default_state in states:
            default_state = "waken"
        channel.with_state(states[default_state], is_default=True)
        for name, state in states.items():
            if name != default_state:
                channel.with_state(state)

        if self._allow_vision:
            channel.import_channels(
                self._vision.as_channel(),
            )

        return channel

    # -- lifecycle ------------------------------------------------------------

    async def bootstrap(self):
        self._mini.__enter__()
        runtime = ChannelCtx.runtime()
        if isinstance(runtime, StatefulChannelRuntimeImpl):
            await runtime.switch_state("waken")

    async def aclose(self):
        self._mini.__exit__(None, None, None)


class ReachyMiniChannelCreator(ChannelCreator):

    def __init__(
            self,
            name: str = "",
            description: str = "",
            default_state: str = "waken",
            allow_vision: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._default_state = default_state
        self._allow_vision = allow_vision

    def factory(self, container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)
        logger = container.force_fetch(LoggerItf)
        if not container.bound(ReachyMini):
            container.set(ReachyMini, ReachyMini())
        mini = container.force_fetch(ReachyMini)
        return MossInReachyMini(
            mini=mini,
            ws=matrix.cell_workspace,
            logger=logger,
            default_state=self._default_state,
            name=self._name,
            description=self._description,
            allow_vision=self._allow_vision,
        ).as_channel()


async def provide_channel(matrix: Matrix) -> None:
    """app 入口: 注入依赖, 组装 Channel, 注册到 Matrix.

    MossInReachyMini 三层模型:
      1. 无副作用类型   — 类定义
      2. 无副作用配置   — 此处: 注入依赖, 组装 Channel
      3. 有副作用实例   — bootstrap/close 生命周期, 连接硬件

    环境变量:
      REACHY_MEDIA_BACKEND: media backend, 默认 no_media (跳过 GStreamer camera)
    """
    import os
    media_backend = os.environ.get('REACHY_MEDIA_BACKEND', 'no_media')
    mini = ReachyMini(media_backend=media_backend)
    reachy = MossInReachyMini(
        mini=mini,
        ws=matrix.cell_workspace,
        logger=matrix.logger,
    )
    await matrix.provide_channel(reachy.as_channel())
