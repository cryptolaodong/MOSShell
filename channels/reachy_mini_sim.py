"""Reachy Mini 模拟 Channel.

在没有真实硬件的情况下，模拟 Reachy Mini 的动作执行。
动作会打印日志而不是驱动真实关节，用于验证 CTML 解析→执行链路。

当连接真实 Reachy Mini 时，替换为 MOSS 自带的
ghoshell_moss_contrib.moss_in_reachy_mini 即可。
"""
from __future__ import annotations

import asyncio
import time


# ── 模拟动作函数 ──────────────────────────────────────

async def head_move(
    x: float = 0,
    y: float = 0,
    z: float = 0,
    roll: float = 0,
    pitch: float = 0,
    yaw: float = 0,
    body_yaw: float = 0,
    duration: float = 0.5,
):
    """Move head to a pose in 6D space.

    Args:
        x: X position [-1.5, +2.5] cm
        y: Y position [-4, +4] cm
        z: Z position [-4, +2.5] cm
        roll: Roll angle [-40, +40] degrees
        pitch: Pitch angle [-40, +40] degrees (negative=down, positive=up)
        yaw: Yaw angle [-60, +60] degrees (positive=left, negative=right)
        body_yaw: Body yaw [-155, +155] degrees
        duration: Duration in seconds
    """
    print(f"  🤖 [head_move] pitch={pitch}° yaw={yaw}° roll={roll}° duration={duration}s")
    await asyncio.sleep(min(duration, 0.1))  # 模拟执行时间（压缩）


async def head_reset(idle_mode: str = "breathing", duration: float = 0.5):
    """Reset head to forward-facing position.

    Args:
        idle_mode: Idle mode after reset - "hold" (still) or "breathing" (recommended)
        duration: Reset duration in seconds
    """
    print(f"  🤖 [head_reset] idle_mode={idle_mode} duration={duration}s")
    await asyncio.sleep(min(duration, 0.1))


async def antennas_move(left: float = 0, right: float = 0, duration: float = 0.3):
    """Move antennas.

    Args:
        left: Left antenna angle [-180, 0] degrees (0=vertical, -180=flat)
        right: Right antenna angle [-180, 0] degrees (0=vertical, -180=flat)
        duration: Duration in seconds
    """
    print(f"  🤖 [antennas_move] left={left}° right={right}° duration={duration}s")
    await asyncio.sleep(min(duration, 0.1))


async def antennas_reset(duration: float = 0.3):
    """Reset antennas to vertical position.

    Args:
        duration: Reset duration in seconds
    """
    print(f"  🤖 [antennas_reset] duration={duration}s")
    await asyncio.sleep(min(duration, 0.1))


async def emotion(name: str = "happy", duration: float = 2.0):
    """Express an emotion through body movement.

    Args:
        name: Emotion name - happy, sad, curious, surprised, sleepy, excited
        duration: Duration in seconds
    """
    print(f"  🤖 [emotion] name={name} duration={duration}s")
    await asyncio.sleep(min(duration, 0.1))


async def dance(name: str = "default", duration: float = 3.0):
    """Perform a dance.

    Args:
        name: Dance name - default, wiggle, nod_beat
        duration: Duration in seconds
    """
    print(f"  🤖 [dance] name={name} duration={duration}s")
    await asyncio.sleep(min(duration, 0.1))


# ── 构建 MOSS Channel ──────────────────────────────────

def build_reachy_mini_sim_channel():
    """构建模拟的 Reachy Mini Channel，注册到 CTML Shell."""
    import sys
    from pathlib import Path
    moss_root = Path(__file__).resolve().parents[1].parent / "MOSShell"
    if str(moss_root / "src") not in sys.path:
        sys.path.insert(0, str(moss_root / "src"))

    from ghoshell_moss import new_prime_channel


    channel = new_prime_channel(
        name="reachy_mini",
        description="Reachy Mini robot body control (simulation mode)",
    )

    # 注册所有动作命令（装饰器方式）
    channel.build.command(name="head_move", blocking=False)(head_move)
    channel.build.command(name="head_reset", blocking=False)(head_reset)
    channel.build.command(name="antennas_move", blocking=False)(antennas_move)
    channel.build.command(name="antennas_reset", blocking=False)(antennas_reset)
    channel.build.command(name="emotion", blocking=False)(emotion)
    channel.build.command(name="dance", blocking=False)(dance)

    return channel


# ── 快速验证 ──────────────────────────────────────
if __name__ == "__main__":
    """验证模拟 Channel 能被 CTML Shell 正确解析和执行."""
    import sys
    from pathlib import Path
    moss_root = Path(__file__).resolve().parents[1].parent / "MOSShell"
    if str(moss_root / "src") not in sys.path:
        sys.path.insert(0, str(moss_root / "src"))

    from ghoshell_moss import new_ctml_shell
    from ghoshell_moss.core.concepts.command import CommandTask

    async def test_ctml_execution():
        # 构建 Shell + Channel
        channel = build_reachy_mini_sim_channel()
        shell = new_ctml_shell()
        shell.main_channel.import_channels(channel)

        # 模拟一段 CTML（小白说"你好"同时挥天线、转头）
        ctml_text = """<_>
<reachy_mini:antennas_move left="-30" right="-30" duration="0.3"/>嘿，你来啦！<reachy_mini:head_move pitch="10" yaw="5" duration="0.4"/>今天过得怎么样？
</_>"""

        print("=" * 50)
        print("  CTML Shell 执行验证")
        print("=" * 50)
        print(f"\n  [输入 CTML]")
        print(f"  {ctml_text}")
        print(f"\n  [执行动作]")

        async with shell:
            interpreter = await shell.interpreter(clear_after_exit=True)
            async with interpreter:
                interpreter.feed(ctml_text)
                interpreter.commit()
                tasks = await asyncio.wait_for(
                    interpreter.wait_tasks(throw=True),
                    timeout=5.0,
                )
                print(f"\n  [完成] 共执行 {len(tasks)} 个命令任务")
                for tid, task in tasks.items():
                    print(f"    - {tid}: {task}")

    asyncio.run(test_ctml_execution())
