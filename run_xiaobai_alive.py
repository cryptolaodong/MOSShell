"""小白 1.0 — 活着的小白.

集成所有能力的完整版本：
- 视觉感知：摄像头检测人脸和表情
- 主动行为：看到人来了主动打招呼（遵守频率限制/安静时段）
- 多轮思考：复杂问题先"想想"再回答
- 语音输出：Mac TTS
- 安全策略：全程在线
- CTML 动作：边说边动

这是当前软件侧最完整的验证版本。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import re
import threading
import time
from datetime import datetime
from pathlib import Path

XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parent
REACHY_ROOT = XIAOBAI_MOSS_ROOT.parent / "Reachy"
MOSS_ROOT = XIAOBAI_MOSS_ROOT.parent / "MOSShell"

sys.path.insert(0, str(REACHY_ROOT / "src"))
sys.path.insert(0, str(MOSS_ROOT / "src"))
sys.path.insert(0, str(XIAOBAI_MOSS_ROOT))

from ghost.xiaobai_ghost import load_env, build_xiaobai_model, load_soul, load_companion_style
from channels.safety_gate import SafetyGateChannel
from channels.face_tracker import classify_emotion
from run_xiaobai_ctml import CTML_INSTRUCTION

from pydantic_ai import Agent

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = str(Path(__file__).parent / "channels" / "face_landmarker.task")


# ── 主动行为系统 ────────────────────────────────────

class ProactiveBehavior:
    """主动行为控制器，基于 Reachy 项目的 proactive_behavior.json."""

    def __init__(self):
        config_path = REACHY_ROOT / "config" / "proactive_behavior.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

        greetings = self._config.get("proactive_greetings", {})
        self._rate_limit_minutes = greetings.get("rate_limit_minutes", 30)
        self._quiet_hours = greetings.get("quiet_hours", {})
        self._max_unanswered = greetings.get("consent_and_interruptibility", {}).get(
            "max_unanswered_greetings_per_day", 3
        )
        self._events = greetings.get("events", {})

        self._last_greeting_time = 0
        self._unanswered_count = 0
        self._face_was_absent = True  # 启动时假设没人

    def is_quiet_hours(self) -> bool:
        """当前是否在安静时段."""
        if not self._quiet_hours.get("enabled", False):
            return False
        now = datetime.now().strftime("%H:%M")
        start = self._quiet_hours.get("start", "21:30")
        end = self._quiet_hours.get("end", "07:00")
        if start <= end:
            return start <= now <= end
        else:  # 跨午夜
            return now >= start or now <= end

    def should_greet(self, face_detected: bool) -> bool:
        """判断是否应该主动打招呼."""
        # 安静时段不打扰
        if self.is_quiet_hours():
            return False

        # 频率限制
        elapsed = time.time() - self._last_greeting_time
        if elapsed < self._rate_limit_minutes * 60:
            return False

        # 未回应次数限制
        if self._unanswered_count >= self._max_unanswered:
            return False

        # 核心逻辑：从"没人"变成"有人"
        if face_detected and self._face_was_absent:
            self._face_was_absent = False
            return True

        if not face_detected:
            self._face_was_absent = True

        return False

    def get_greeting(self, event: str = "child_entered") -> str:
        """从预设问候语中随机选一条."""
        greetings = self._events.get(event, [])
        if greetings:
            return random.choice(greetings)
        return "嘿，你来啦。"

    def record_greeting(self, got_response: bool = False):
        """记录一次打招呼."""
        self._last_greeting_time = time.time()
        if not got_response:
            self._unanswered_count += 1
        else:
            self._unanswered_count = 0


# ── 视觉追踪（后台线程）────────────────────────────

class VisionThread:
    def __init__(self):
        self._emotion = "neutral"
        self._confidence = 0.0
        self._face_detected = False
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def emotion(self): return self._emotion
    @property
    def confidence(self): return self._confidence
    @property
    def face_detected(self): return self._face_detected

    def _loop(self):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [视觉] 摄像头无法打开")
            return

        while self._running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.face_landmarks:
                self._face_detected = True
                self._emotion, self._confidence = classify_emotion(result.face_landmarks)
            else:
                self._face_detected = False
                self._emotion = "no_face"
            time.sleep(0.2)

        cap.release()
        landmarker.close()


# ── 工具函数 ──────────────────────────────────────

def speak(text: str):
    if text.strip():
        subprocess.run(["say", "-v", "Tingting", "-r", "200", text], check=False)

def action_sound():
    subprocess.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def thinking_sound():
    """思考时的音效."""
    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def strip_ctml(text: str) -> str:
    text = re.sub(r'</?_[^>]*>', '', text)
    text = re.sub(r'<[^>]+/>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_actions(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'<reachy_mini:(\w+)', text)]


# ── 多轮思考 ──────────────────────────────────────

def needs_deeper_thinking(child_text: str, safety_level: str) -> bool:
    """判断这个问题是否需要多想一会儿.

    复杂问题/敏感话题 → 小白先犹豫一下再回答。
    """
    # 红灯话题需要思考
    if safety_level == "red":
        return True
    # 包含疑问 + 较长的输入可能需要思考
    if len(child_text) > 20 and any(w in child_text for w in ["为什么", "怎么办", "是不是", "到底", "真的吗"]):
        return True
    return False


# ── 主循环 ────────────────────────────────────────

async def main():
    load_env()

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║     小白 1.0 — 活着的小白            ║")
    print("  ║  视觉 + 主动 + 思考 + 语音 + 安全    ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    model = build_xiaobai_model()
    safety_gate = SafetyGateChannel()
    proactive = ProactiveBehavior()
    print("  ✓ 大脑 + 安全 + 主动行为")

    # 视觉
    vis = VisionThread()
    vis.start()
    time.sleep(1)
    print("  ✓ 视觉感知（摄像头后台运行）")

    # Agent
    soul = load_soul()
    style = load_companion_style()
    system_parts = [soul, CTML_INSTRUCTION]
    if style:
        system_parts.append(f"\n## 陪伴风格\n\n{style}")
    system_parts.append("""
## 视觉感知
你能看到陶陶的表情。根据他的情绪自然调整回应风格和动作幅度。
不要每次都报告你看到了什么，自然地感知就好。

## 思考表达
当你需要想一想时，可以先用一个犹豫的动作（歪头、天线轻转），
然后再回答。这让陶陶知道你在认真想他的问题。
""")
    agent = Agent(model=model, system_prompt="\n".join(system_parts))
    print("  ✓ 小白 Ghost 完整就绪")
    print()

    # 主动打招呼
    if vis.face_detected:
        greeting = proactive.get_greeting("child_entered")
        action_sound()
        print(f"  小白（主动）: {greeting}")
        speak(greeting)
        proactive.record_greeting()
    else:
        print("  小白在等人来...")

    print()
    print("  输入中文跟小白聊天 | 输入 quit 退出 | 什么都不输直接回车=等待主动行为")
    print()

    history = []

    try:
        while True:
            try:
                child_text = input("  陶陶: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            # 空输入：检查主动行为
            if not child_text:
                if proactive.should_greet(vis.face_detected):
                    greeting = proactive.get_greeting("child_entered")
                    action_sound()
                    print(f"  小白（主动）: {greeting}")
                    speak(greeting)
                    proactive.record_greeting()
                else:
                    status = f"👁 {'有人' if vis.face_detected else '没人'} | 😊 {vis.emotion}"
                    print(f"  [{status}] 小白安静地等着...")
                print()
                continue

            if child_text.lower() in ("quit", "exit", "q"):
                break

            # 有回应，重置未回应计数
            proactive.record_greeting(got_response=True)

            # 视觉上下文
            emotion_zh = {"happy": "开心", "sad": "难过", "surprised": "惊讶",
                          "frustrated": "不开心", "neutral": "平静", "no_face": "没看到脸"}
            vis_ctx = f"陶陶表情：{emotion_zh.get(vis.emotion, '未知')}" if vis.face_detected else "没有检测到人脸"

            # 安全评估
            evaluation = safety_gate.evaluate_input(child_text)
            redacted = safety_gate.redact(child_text)
            level = evaluation.get("safety_level", "?")
            level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "⚪")
            print(f"  [{level_icon} {level}] [👁 {vis.emotion}]")

            # 多轮思考
            if needs_deeper_thinking(child_text, level):
                thinking_sound()
                print("  小白（在想...）")
                await asyncio.sleep(1.0)  # 真正停顿一秒

            # 构造输入
            augmented = f"[{vis_ctx}]\n\n陶陶说：{redacted}"

            try:
                result = await agent.run(
                    user_prompt=augmented,
                    message_history=history if history else None,
                )
                ctml_output = result.output
                history.extend(result.new_messages())

                speech = strip_ctml(ctml_output)
                actions = parse_actions(ctml_output)

                if actions:
                    action_sound()
                    await asyncio.sleep(0.2)

                print(f"  小白: {speech}")
                speak(speech)
                print()

            except Exception as e:
                print(f"  [错误] {e}\n")

    finally:
        vis.stop()
        speak("再见啦，明天见。")
        print("\n  小白: 再见啦，明天见。")


if __name__ == "__main__":
    asyncio.run(main())
