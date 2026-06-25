"""小白 1.0 — 可感知版本.

你能听到小白说话（Mac TTS），动作有系统音效提示。
这是给人类验证用的——用耳朵确认系统是真的在工作。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import re
from pathlib import Path

XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parent
REACHY_ROOT = XIAOBAI_MOSS_ROOT.parent / "Reachy"
MOSS_ROOT = XIAOBAI_MOSS_ROOT.parent / "MOSShell"

sys.path.insert(0, str(REACHY_ROOT / "src"))
sys.path.insert(0, str(MOSS_ROOT / "src"))
sys.path.insert(0, str(XIAOBAI_MOSS_ROOT))

from ghost.xiaobai_ghost import load_env, build_xiaobai_model, load_soul, load_companion_style
from channels.safety_gate import SafetyGateChannel
from run_xiaobai_ctml import CTML_INSTRUCTION

from pydantic_ai import Agent


def speak(text: str):
    """用 Mac TTS 播放中文语音."""
    if text.strip():
        subprocess.run(["say", "-v", "Tingting", "-r", "200", text], check=False)


def action_sound():
    """播放一个短音效表示动作正在执行."""
    # Mac 系统自带音效
    subprocess.Popen(
        ["afplay", "/System/Library/Sounds/Pop.aiff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def strip_ctml_to_speech(ctml_text: str) -> str:
    """从 CTML 提取纯语音文本."""
    text = re.sub(r'</?_[^>]*>', '', ctml_text)
    text = re.sub(r'<[^>]+/>', '', text)
    text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_actions(ctml_text: str) -> list[dict]:
    """解析 CTML 中的动作."""
    actions = []
    pattern = r'<reachy_mini:(\w+)\s+([^/]*)/>'
    for match in re.finditer(pattern, ctml_text):
        command = match.group(1)
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2)))
        actions.append({"command": command, "attrs": attrs})
    return actions


async def main():
    load_env()

    print()
    print("  小白 1.0 — 可感知验证")
    print("  你会听到小白说话，动作有音效")
    print("  输入中文跟小白聊天，输入 quit 退出")
    print()

    model = build_xiaobai_model()
    safety_gate = SafetyGateChannel()

    soul = load_soul()
    style = load_companion_style()
    system_parts = [soul, CTML_INSTRUCTION]
    if style:
        system_parts.append(f"\n## 陪伴风格规则\n\n{style}")

    agent = Agent(model=model, system_prompt="\n".join(system_parts))

    # 开场白——小白主动说一句
    speak("你好呀，我是小白，你的小伙伴。")
    print("  小白: 你好呀，我是小白，你的小伙伴。")
    print()

    history = []

    while True:
        try:
            child_text = input("  陶陶: ").strip()
        except (EOFError, KeyboardInterrupt):
            speak("再见啦")
            print("\n  小白: 再见啦")
            break

        if not child_text:
            continue
        if child_text.lower() in ("quit", "exit", "q"):
            speak("再见啦，下次再聊")
            print("  小白: 再见啦，下次再聊")
            break

        # 安全评估
        evaluation = safety_gate.evaluate_input(child_text)
        redacted_text = safety_gate.redact(child_text)
        level = evaluation.get("safety_level", "?")

        try:
            # LLM 生成 CTML
            result = await agent.run(
                user_prompt=redacted_text,
                message_history=history if history else None,
            )
            ctml_output = result.output
            history.extend(result.new_messages())

            # 解析
            speech = strip_ctml_to_speech(ctml_output)
            actions = parse_actions(ctml_output)

            # 执行动作音效
            if actions:
                action_sound()
                action_names = [a["command"] for a in actions]
                print(f"  [动作: {', '.join(action_names)}]")
                await asyncio.sleep(0.3)  # 短暂延迟让音效先响

            # 说话
            print(f"  小白: {speech}")
            speak(speech)
            print()

        except Exception as e:
            print(f"  [错误] {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
