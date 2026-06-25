"""小白 1.0 — 真实硬件回合制版本.

goto_target 同步动作 + OpenAI gpt-4o-mini-tts 有感情语音。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parent
REACHY_ROOT = XIAOBAI_MOSS_ROOT.parent / "Reachy"
MOSS_ROOT = XIAOBAI_MOSS_ROOT.parent / "MOSShell"

sys.path.insert(0, str(REACHY_ROOT / "src"))
sys.path.insert(0, str(MOSS_ROOT / "src"))
sys.path.insert(0, str(XIAOBAI_MOSS_ROOT))

from ghost.xiaobai_ghost import load_env, build_xiaobai_model, load_soul, load_companion_style
from channels.safety_gate import SafetyGateChannel
from pydantic_ai import Agent
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


# ── 加载 Reachy 项目的 .env ───────────────────────────

def load_reachy_env():
    """加载 Reachy 项目的环境变量（含 OpenAI TTS 配置）。"""
    for env_path in (REACHY_ROOT / ".env", REACHY_ROOT / ".env.local"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


# ── 有感情的 TTS（OpenAI gpt-4o-mini-tts）─────────────

def speak(text: str):
    """用 OpenAI TTS 合成有感情的语音并播放。失败时降级到 Mac say。"""
    if not text.strip():
        return

    api_key = os.environ.get("AUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("AUDIO_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("AUDIO_TTS_MODEL", "gpt-4o-mini-tts")
    voice = os.environ.get("AUDIO_TTS_VOICE", "ash")
    instructions = os.environ.get("AUDIO_TTS_INSTRUCTIONS", "")

    if not api_key:
        # 降级到 Mac say
        subprocess.run(["say", "-v", "Tingting", "-r", "190", text], check=False)
        return

    # 构建 TTS 请求
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "mp3",
    }
    if instructions:
        payload["instructions"] = instructions

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/audio/speech",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            audio_data = resp.read()

        # 写临时文件播放
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        # 用 afplay 播放
        subprocess.run(["afplay", tmp_path], check=False)
        os.unlink(tmp_path)

    except Exception as e:
        print(f"  [TTS降级] {e}")
        subprocess.run(["say", "-v", "Tingting", "-r", "190", text], check=False)


# ── 控制器 ──────────────────────────────────────────

class SafeReachyController:
    """同步阻塞式控制器。goto_target 会等动作做完才返回。

    动作设计原则：
    - 所有动作用 minjerk 插值（默认），平滑无抖动
    - duration 1.5-2.5 秒，不急不慢
    - 回正位置带微小随机偏移，避免机械感
    - pitch 负值=抬头，正值=低头（已验证方向）
    """

    def __init__(self):
        self._mini = ReachyMini(media_backend='no_media')

    def connect(self):
        self._mini.__enter__()
        self._mini.enable_motors()
        self._mini.wake_up()
        time.sleep(2)

    def disconnect(self):
        self._mini.__exit__(None, None, None)

    def _natural_rest(self):
        """回到自然休息姿态。"""
        import random
        r = lambda: random.uniform(-2, 2)
        self._mini.goto_target(
            head=create_head_pose(pitch=r(), yaw=r(), roll=r()),
            duration=1.8
        )
        time.sleep(0.3)

    def reset(self):
        self._natural_rest()

    def gesture(self, name: str):
        """执行手势。pitch 负=抬头，正=低头。"""
        if name == "greeting":
            self._mini.goto_target(head=create_head_pose(pitch=-15, yaw=5), duration=1.8)
            time.sleep(0.3)

        elif name == "nod":
            self._mini.goto_target(head=create_head_pose(pitch=18), duration=1.0)
            time.sleep(0.2)
            self._mini.goto_target(head=create_head_pose(pitch=-5), duration=0.8)
            time.sleep(0.2)
            self._natural_rest()

        elif name == "tilt_curious":
            self._mini.goto_target(head=create_head_pose(roll=15, pitch=-8, yaw=5), duration=1.8)
            time.sleep(0.3)

        elif name == "look_up_excited":
            self._mini.goto_target(head=create_head_pose(pitch=-20, yaw=-3), duration=1.5)
            time.sleep(0.3)

        elif name == "sad":
            self._mini.goto_target(head=create_head_pose(pitch=22, roll=-3), duration=2.5)
            time.sleep(0.3)

        elif name == "thinking":
            self._mini.goto_target(head=create_head_pose(roll=12, yaw=10, pitch=-3), duration=1.8)
            time.sleep(0.8)

        elif name == "turn_look":
            self._mini.goto_target(head=create_head_pose(yaw=25, pitch=-5), duration=1.8)
            time.sleep(0.3)

        elif name == "goodbye":
            self._mini.goto_target(head=create_head_pose(pitch=20, roll=-5), duration=2.5)
            time.sleep(0.5)


# ── 文本处理 ──────────────────────────────────────

def strip_ctml(text: str) -> str:
    text = re.sub(r'</?_[^>]*>', '', text)
    text = re.sub(r'<[^>]+/>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_gesture(text: str) -> str:
    match = re.search(r'<xiaobai:gesture name="([^"]+)"', text)
    return match.group(1) if match else "none"


# ── CTML 指令 ──────────────────────────────────────

SYSTEM_PROMPT = """【格式要求 - 必须严格遵守】
你的每一条回复都必须用以下格式：

<_>
<xiaobai:gesture name="手势名"/>回复文字
</_>

可用手势：greeting, nod, tilt_curious, look_up_excited, sad, thinking, turn_look, none

手势含义：
- greeting: 抬头看人（打招呼）
- nod: 点头（理解了）
- tilt_curious: 好奇歪头（想了解更多）
- look_up_excited: 兴奋抬头（很棒的事）
- sad: 低头（共情难过）
- thinking: 歪头想（思考）
- turn_look: 转头（注意到什么）
- none: 不动（普通对话）

你是小白，6岁孩子陶陶的机器人伙伴。温暖、好奇、有耐心。
1-3句话，最多一个问号。先接住孩子的话再引导。
孩子难过时先陪伴，建议找爸爸妈妈或老师。
"""


# ── 主循环 ──────────────────────────────────────────

async def main():
    load_env()
    load_reachy_env()

    print()
    print("  小白 1.0")
    print()

    controller = SafeReachyController()
    controller.connect()
    print("  ✓ Reachy Mini 就绪")

    model = build_xiaobai_model()
    safety_gate = SafetyGateChannel()
    print(f"  ✓ 大脑: {os.environ.get('XIAOBAI_MODEL', 'qwen-plus')}")

    tts_model = os.environ.get("AUDIO_TTS_MODEL", "?")
    tts_voice = os.environ.get("AUDIO_TTS_VOICE", "?")
    print(f"  ✓ 语音: {tts_model} / {tts_voice}")

    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)
    print("  ✓ Ghost")
    print()

    # 开场
    controller.gesture("greeting")
    speak("你好呀，我是小白。")
    controller.reset()
    print("  小白: 你好呀，我是小白。")
    print()
    print("  跟小白聊天 | quit 退出")
    print()

    history = []

    try:
        while True:
            try:
                child_text = input("  陶陶: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not child_text:
                continue
            if child_text.lower() in ("quit", "exit", "q"):
                break

            evaluation = safety_gate.evaluate_input(child_text)
            redacted = safety_gate.redact(child_text)
            level = evaluation.get("safety_level", "?")
            level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "⚪")
            print(f"  [{level_icon}]")

            try:
                result = await agent.run(
                    user_prompt=redacted,
                    message_history=history if history else None,
                )
                ctml_output = result.output
                history.extend(result.new_messages())

                speech = strip_ctml(ctml_output)
                gesture = extract_gesture(ctml_output)

                # 先动
                if gesture != "none":
                    print(f"  [动作: {gesture}]")
                    controller.gesture(gesture)

                # 再说（有感情的语音）
                print(f"  小白: {speech}")
                speak(speech)

                # 部分手势说完回正
                if gesture in ("nod", "look_up_excited", "turn_look"):
                    controller.reset()

                print()

            except Exception as e:
                print(f"  [错误] {e}\n")

    finally:
        controller.gesture("goodbye")
        speak("再见啦")
        time.sleep(1)
        controller.disconnect()
        print("\n  再见（已断开）")


if __name__ == "__main__":
    asyncio.run(main())
