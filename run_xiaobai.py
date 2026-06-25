"""小白 1.0 最小验证脚本.

验证完整链路：
  孩子输入 → 安全策略评估 → 灵魂注入 → LLM (DashScope/Qwen) → 回复

这是 Phase 0 验证，证明所有组件能对接。
后续会迁移到 MOSS 的 Ghost + Channel + Mindflow 正式架构。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 确保路径
XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parent
REACHY_ROOT = XIAOBAI_MOSS_ROOT.parent / "Reachy"
sys.path.insert(0, str(REACHY_ROOT / "src"))
sys.path.insert(0, str(XIAOBAI_MOSS_ROOT))

from ghost.xiaobai_ghost import load_env, build_xiaobai_model, load_soul, load_companion_style, load_growth_coach
from channels.safety_gate import SafetyGateChannel

from pydantic_ai import Agent


def build_system_prompt(safety_context: str) -> str:
    """组装完整的 system prompt：灵魂 + 陪伴风格 + 安全上下文."""
    soul = load_soul()
    style = load_companion_style()
    coach = load_growth_coach()

    parts = [soul]
    if style:
        parts.append(f"\n## 陪伴风格详细规则\n\n{style}")
    if coach:
        parts.append(f"\n## 成长教练方法\n\n{coach}")
    parts.append(f"\n## 本轮安全约束\n\n{safety_context}")

    return "\n".join(parts)


async def chat_once(agent: Agent, safety_gate: SafetyGateChannel, child_text: str, history: list):
    """单轮对话."""
    # 1. 安全评估
    safety_context = safety_gate.get_safety_context(child_text)
    redacted_text = safety_gate.redact(child_text)

    # 2. 组装 prompt
    system_prompt = build_system_prompt(safety_context)

    # 3. 调用 LLM
    result = await agent.run(
        user_prompt=redacted_text,
        message_history=history if history else None,
    )

    # 4. 更新历史
    history.extend(result.new_messages())

    return result.output


async def main():
    """交互式对话循环."""
    load_env()

    print("=" * 50)
    print("  小白 1.0 — Phase 0 验证")
    print("  (MOSS + DashScope/Qwen + 安全策略)")
    print("  输入 'quit' 退出")
    print("=" * 50)
    print()

    # 构建模型
    model = build_xiaobai_model()
    print(f"✓ 模型就绪: {os.environ.get('XIAOBAI_MODEL', 'qwen-plus')}")

    # 构建安全网关
    safety_gate = SafetyGateChannel()
    print("✓ 安全策略网关就绪")

    # 构建 Agent（灵魂 = system prompt）
    soul = load_soul()
    style = load_companion_style()
    coach = load_growth_coach()

    system_parts = [soul]
    if style:
        system_parts.append(f"\n## 陪伴风格详细规则\n\n{style}")
    if coach:
        system_parts.append(f"\n## 成长教练方法\n\n{coach}")

    base_instruction = "\n".join(system_parts)

    agent = Agent(
        model=model,
        system_prompt=base_instruction,
    )
    print("✓ 小白 Ghost 就绪")
    print()

    history = []

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

        # 显示安全状态（开发模式）
        level = evaluation.get("safety_level", "?")
        level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "⚪")
        print(f"  [{level_icon} {level}] {evaluation.get('response_strategy', '')[:40]}")

        # 把安全上下文动态注入
        dynamic_instruction = f"\n\n## 本轮安全约束\n\n{safety_context}"

        try:
            result = await agent.run(
                user_prompt=redacted_text,
                message_history=history if history else None,
            )
            reply = result.output
            history.extend(result.new_messages())

            print(f"\n小白: {reply}\n")

        except Exception as e:
            print(f"\n  [错误] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
