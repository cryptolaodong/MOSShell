"""小白 1.0 — 完整闭环验证.

完整链路：
  孩子输入 → 安全评估 → LLM 输出 CTML → CTML Shell 解析 → 动作真正执行

这是 Phase 2 的核心验证：证明从"孩子说话"到"机器人动起来"的完整链路跑通。
模拟模式下动作打印日志，换成真实 Reachy Mini 就是控制关节。
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
from run_xiaobai_ctml import CTML_INSTRUCTION, strip_ctml_to_speech

from pydantic_ai import Agent
from ghoshell_moss import new_ctml_shell


async def execute_ctml(shell, ctml_text: str):
    """用 CTML Shell 执行一段 CTML 文本."""
    interpreter = await shell.interpreter(clear_after_exit=True)
    async with interpreter:
        interpreter.feed(ctml_text)
        interpreter.commit()
        try:
            tasks = await asyncio.wait_for(
                interpreter.wait_tasks(throw=True),
                timeout=10.0,
            )
            return tasks
        except asyncio.TimeoutError:
            print("  [警告] CTML 执行超时")
            return {}
        except Exception as e:
            print(f"  [警告] CTML 执行异常: {e}")
            return {}


async def main():
    """完整闭环：输入 → 安全评估 → LLM(CTML) → CTML Shell 执行动作."""
    load_env()

    print("=" * 60)
    print("  小白 1.0 — Phase 2: 完整闭环")
    print("  孩子说话 → 安全评估 → LLM → CTML → 动作执行")
    print("  (模拟模式：动作打印日志)")
    print("  输入 'quit' 退出")
    print("=" * 60)
    print()

    # 1. 构建 LLM
    model = build_xiaobai_model()
    print(f"  ✓ 大脑就绪: {os.environ.get('XIAOBAI_MODEL', 'qwen-plus')}")

    # 2. 构建安全网关
    safety_gate = SafetyGateChannel()
    print("  ✓ 安全策略网关就绪")

    # 3. 构建 CTML Shell + Reachy Mini Channel
    reachy_channel = build_reachy_mini_sim_channel()
    shell = new_ctml_shell()
    shell.main_channel.import_channels(reachy_channel)
    print("  ✓ CTML Shell + Reachy Mini (模拟) 就绪")

    # 4. 构建 Ghost Agent
    soul = load_soul()
    style = load_companion_style()
    system_parts = [soul, CTML_INSTRUCTION]
    if style:
        system_parts.append(f"\n## 陪伴风格规则\n\n{style}")
    base_instruction = "\n".join(system_parts)

    agent = Agent(model=model, system_prompt=base_instruction)
    print("  ✓ 小白 Ghost 就绪")
    print()
    print("  一切就绪。小白在等陶陶说话。")
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

            try:
                # LLM 生成 CTML
                result = await agent.run(
                    user_prompt=redacted_text,
                    message_history=history if history else None,
                )
                ctml_output = result.output
                history.extend(result.new_messages())

                # 提取语音文本
                speech = strip_ctml_to_speech(ctml_output)
                print(f"\n  小白说: {speech}")

                # CTML Shell 执行动作
                print(f"  小白动:")
                tasks = await execute_ctml(shell, ctml_output)

                if not tasks:
                    print("    (无动作)")
                print()

            except Exception as e:
                print(f"\n  [错误] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
