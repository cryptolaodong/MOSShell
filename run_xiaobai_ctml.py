"""小白 1.0 — CTML 验证脚本.

验证模型能输出 CTML 格式（边说边动）：
  孩子输入 → 安全评估 → LLM 输出 CTML（语言+动作混合）→ 解析验证

这是 Phase 1 的关键验证：证明大模型能按照 CTML 格式
把"说什么"和"做什么动作"编织在一起输出。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parent
REACHY_ROOT = XIAOBAI_MOSS_ROOT.parent / "Reachy"
sys.path.insert(0, str(REACHY_ROOT / "src"))
sys.path.insert(0, str(XIAOBAI_MOSS_ROOT))

from ghost.xiaobai_ghost import load_env, build_xiaobai_model, load_soul, load_companion_style
from channels.safety_gate import SafetyGateChannel
from pydantic_ai import Agent


# CTML 指令——教模型如何输出"边说边动"的格式
CTML_INSTRUCTION = """
## 输出格式：CTML（流式控制语言）

你的回复必须使用 CTML 格式。CTML 让你可以在说话的同时做动作。

基本规则：
1. 所有回复包裹在 `<_> ... </_>` 作用域内
2. 文字直接写在作用域里，这些文字会被转成语音播放
3. 动作用 XML 标签插入，标签闭合的瞬间动作就开始执行
4. 动作和文字可以交织——你在说某句话的中间插入动作，动作就在那个时间点执行

### 可用动作

```
<reachy_mini:head_move x="0" y="0" z="0" roll="0" pitch="0" yaw="0" body_yaw="0" duration="0.5"/>
```
头部运动。参数范围：
- x: [-1.5, +2.5] cm，y: [-4, +4] cm，z: [-4, +2.5] cm
- roll: [-40, +40]°，pitch: [-40, +40]°（负=低头，正=抬头）
- yaw: [-60, +60]°（正=左转，负=右转）
- body_yaw: [-155, +155]°
- duration: 动作时长（秒）

```
<reachy_mini:head_reset idle_mode="breathing" duration="0.5"/>
```
头部复位，面朝正前方。idle_mode 可选 "hold"（静止）或 "breathing"（呼吸动画，推荐）。

```
<reachy_mini:antennas_move left="0" right="0" duration="0.3"/>
```
天线动作。left/right 范围: [-180, 0]°，0=竖直，-180=完全放平。

```
<reachy_mini:antennas_reset duration="0.3"/>
```
天线复位（竖直）。

```
<reachy_mini:emotion name="happy" duration="2.0"/>
```
情绪表达动作。name 可选: happy, sad, curious, surprised, sleepy, excited

```
<reachy_mini:dance name="default" duration="3.0"/>
```
跳舞。name 可选: default, wiggle, nod_beat

### 示例

孩子说"小白你好"，你可以回：
```
<_>
<reachy_mini:antennas_move left="-30" right="-30" duration="0.3"/>嘿，你来啦！<reachy_mini:head_move pitch="10" yaw="5" duration="0.4"/>今天过得怎么样？
</_>
```
（天线先轻轻动一下表示注意到了 → 说"嘿，你来啦！" → 头微微抬起偏一点表示好奇 → 问"今天过得怎么样？"）

孩子说"我有点难过"，你可以回：
```
<_>
<reachy_mini:head_move pitch="-5" duration="0.4"/>嗯，我听到了。<reachy_mini:antennas_move left="-90" right="-90" duration="0.5"/>想跟我说说吗？
</_>
```
（头微微低下来 → 说"嗯，我听到了。" → 天线慢慢放平表示安静陪伴 → 问"想跟我说说吗？"）

### 重要原则

- 动作要自然，不要每句话都加动作，关键时刻用就好
- 动作幅度要小而温和（记住你面对的是 6 岁小孩）
- 情绪匹配：开心时天线竖起/头抬高，安静时天线放平/头微低
- 不要在孩子害怕或难过时做突然的大幅度动作
- 安静陪伴时可以只用 head_reset idle_mode="breathing"，让身体自然呼吸
"""


def build_ctml_system_prompt(safety_context: str) -> str:
    """组装带 CTML 指令的 system prompt."""
    soul = load_soul()
    style = load_companion_style()

    parts = [soul, CTML_INSTRUCTION]
    if style:
        parts.append(f"\n## 陪伴风格规则\n\n{style}")
    parts.append(f"\n## 本轮安全约束\n\n{safety_context}")

    return "\n".join(parts)


def parse_ctml_actions(ctml_text: str) -> list[dict]:
    """简单解析 CTML 中的动作标签（验证用）."""
    import re
    actions = []
    # 匹配 <reachy_mini:command attr="value" ... />
    pattern = r'<reachy_mini:(\w+)\s+([^/]*)/>'
    for match in re.finditer(pattern, ctml_text):
        command = match.group(1)
        attrs_str = match.group(2)
        # 解析属性
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))
        actions.append({"command": command, "attrs": attrs, "position": match.start()})
    return actions


def strip_ctml_to_speech(ctml_text: str) -> str:
    """从 CTML 中提取纯语音文本（去掉标签和作用域）."""
    import re
    # 去掉 <_> </_> 作用域标签
    text = re.sub(r'</?_[^>]*>', '', ctml_text)
    # 去掉所有 XML 标签
    text = re.sub(r'<[^>]+/>', '', text)
    text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def main():
    """交互式 CTML 对话验证."""
    load_env()

    print("=" * 60)
    print("  小白 1.0 — Phase 1: CTML 验证")
    print("  模型输出 CTML = 语言 + 动作 编织在一起")
    print("  输入 'quit' 退出")
    print("=" * 60)
    print()

    model = build_xiaobai_model()
    print(f"✓ 模型就绪: {os.environ.get('XIAOBAI_MODEL', 'qwen-plus')}")

    safety_gate = SafetyGateChannel()
    print("✓ 安全策略网关就绪")

    # 用带 CTML 指令的 system prompt 构建 agent
    base_instruction = build_ctml_system_prompt("")
    agent = Agent(model=model, system_prompt=base_instruction)
    print("✓ 小白 Ghost (CTML 模式) 就绪")
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

        level = evaluation.get("safety_level", "?")
        level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "⚪")
        print(f"  [{level_icon} {level}]")

        try:
            result = await agent.run(
                user_prompt=redacted_text,
                message_history=history if history else None,
            )
            ctml_output = result.output
            history.extend(result.new_messages())

            # 解析动作
            actions = parse_ctml_actions(ctml_output)
            speech = strip_ctml_to_speech(ctml_output)

            # 显示结果
            print(f"\n  [CTML 原始输出]")
            print(f"  {ctml_output}")
            print()
            print(f"  [语音文本] {speech}")
            if actions:
                print(f"  [动作指令] {len(actions)} 个:")
                for i, act in enumerate(actions, 1):
                    print(f"    {i}. {act['command']}({act['attrs']})")
            else:
                print(f"  [动作指令] 无（纯语音回复）")
            print()

        except Exception as e:
            print(f"\n  [错误] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
