"""小白 1.0 — 流式执行版.

核心体验升级：模型边生成 token 边执行动作。
不是等整段 CTML 生成完再执行，而是 token 一到就解析、标签一闭合就动。

这才是"活着"的感觉——模型还没说完，小白已经在动了。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parent
REACHY_ROOT = XIAOBAI_MOSS_ROOT.parent / "Reachy"
MOSS_ROOT = XIAOBAI_MOSS_ROOT.parent / "MOSShell"

sys.path.insert(0, str(REACHY_ROOT / "src"))
sys.path.insert(0, str(MOSS_ROOT / "src"))
sys.path.insert(0, str(XIAOBAI_MOSS_ROOT))

from ghost.xiaobai_ghost import load_env, build_xiaobai_model, load_soul, load_companion_style
from channels.safety_gate import SafetyGateChannel
from channels.reachy_mini_sim import build_reachy_mini_sim_channel
from run_xiaobai_ctml import CTML_INSTRUCTION

from pydantic_ai import Agent
from ghoshell_moss import new_ctml_shell


async def stream_and_execute(agent: Agent, shell, child_text: str, history: list):
    """流式生成 + 实时执行.

    模型每吐出一段 token，立即 feed 给 CTML Shell。
    CTML 标签一闭合，对应动作立即执行。
    文字内容实时打印（模拟 TTS 播放）。
    """
    # 启动 CTML 解释器
    interpreter = await shell.interpreter(clear_after_exit=True)

    async with interpreter:
        # 流式调用 LLM
        async with agent.run_stream(
            user_prompt=child_text,
            message_history=history if history else None,
        ) as stream:
            full_output = ""

            async for delta in stream.stream_text(delta=True):
                # 每个 delta 立即喂给 CTML Shell
                interpreter.feed(delta)
                full_output += delta

                # 实时打印纯文字部分（去掉 XML 标签）
                # 简单处理：只打印不包含 < 或 > 的片段
                clean = delta.replace("</_>", "").replace("<_>", "")
                if "<" not in clean and ">" not in clean and clean.strip():
                    print(clean, end="", flush=True)

            # 所有 token 生成完毕，通知解释器
            interpreter.commit()

            # 等待所有动作执行完
            try:
                tasks = await asyncio.wait_for(
                    interpreter.wait_tasks(throw=True),
                    timeout=10.0,
                )
            except (asyncio.TimeoutError, Exception):
                tasks = {}

        # 更新对话历史
        history.extend(stream.new_messages())

    return full_output, tasks


async def main():
    """流式交互循环."""
    load_env()

    print("=" * 60)
    print("  小白 1.0 — 流式执行")
    print("  模型边想边说边动，token 级实时响应")
    print("  输入 'quit' 退出")
    print("=" * 60)
    print()

    # 构建所有组件
    model = build_xiaobai_model()
    print(f"  ✓ 大脑: {os.environ.get('XIAOBAI_MODEL', 'qwen-plus')}")

    safety_gate = SafetyGateChannel()
    print("  ✓ 安全策略网关")

    reachy_channel = build_reachy_mini_sim_channel()
    shell = new_ctml_shell()
    shell.main_channel.import_channels(reachy_channel)
    print("  ✓ CTML Shell + Reachy Mini (模拟)")

    soul = load_soul()
    style = load_companion_style()
    system_parts = [soul, CTML_INSTRUCTION]
    if style:
        system_parts.append(f"\n## 陪伴风格规则\n\n{style}")
    base_instruction = "\n".join(system_parts)

    agent = Agent(model=model, system_prompt=base_instruction)
    print("  ✓ 小白 Ghost")
    print()
    print("  小白醒着，在等陶陶。")
    print()

    history = []

    async with shell:
        while True:
            try:
                child_text = input("陶陶: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not child_text:
                continue
            if child_text.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            # 安全评估
            safety_context = safety_gate.get_safety_context(child_text)
            redacted_text = safety_gate.redact(child_text)
            evaluation = safety_gate.evaluate_input(child_text)

            level = evaluation.get("safety_level", "?")
            level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "⚪")
            print(f"  [{level_icon} {level}]")
            print()
            print("  小白: ", end="", flush=True)

            try:
                full_output, tasks = await stream_and_execute(
                    agent, shell, redacted_text, history
                )
                print()  # 换行

                # 统计动作数
                action_count = sum(
                    1 for t in tasks.values()
                    if hasattr(t, '_chan') and 'reachy_mini' in str(getattr(t, '_chan', ''))
                )
                if tasks:
                    print(f"  (执行了 {len(tasks)} 个任务)")
                print()

            except Exception as e:
                print(f"\n  [错误] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
