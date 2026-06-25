"""小白 1.0 — 视觉感知版.

摄像头持续检测表情，对话时小白知道你现在什么情绪。
你能听到小白说话，同时看到它检测到了你的表情。

验证点：
- 你笑着说话 → 小白回应更活泼
- 你皱眉说话 → 小白回应更温柔
- 小白会在回复中提到它"看到"的东西
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import re
import threading
import time
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


class EmotionTracker:
    """后台线程持续检测表情，主线程随时读取最新情绪."""

    def __init__(self):
        self._emotion = "neutral"
        self._confidence = 0.0
        self._running = False
        self._thread = None
        self._face_detected = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def emotion(self) -> str:
        return self._emotion

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def face_detected(self) -> bool:
        return self._face_detected

    def get_context(self) -> str:
        """生成给 Ghost 的视觉上下文."""
        if not self._face_detected:
            return "视觉感知：当前没有检测到人脸。"

        emotion_zh = {
            "happy": "开心（在笑）",
            "sad": "有点难过（嘴角下拉）",
            "surprised": "很惊讶（嘴巴张大）",
            "frustrated": "有点不开心（皱眉）",
            "neutral": "表情平静",
        }
        desc = emotion_zh.get(self._emotion, "表情平静")
        return f"视觉感知：陶陶现在的表情是{desc}，置信度{self._confidence:.0%}。"

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
            print("  [视觉] 无法打开摄像头")
            return

        while self._running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                self._face_detected = True
                self._emotion, self._confidence = classify_emotion(result.face_landmarks)
            else:
                self._face_detected = False
                self._emotion = "no_face"
                self._confidence = 0.0

            time.sleep(0.2)  # 5fps 够用了

        cap.release()
        landmarker.close()


def speak(text: str):
    """Mac TTS."""
    if text.strip():
        subprocess.run(["say", "-v", "Tingting", "-r", "200", text], check=False)


def action_sound():
    """动作音效."""
    subprocess.Popen(
        ["afplay", "/System/Library/Sounds/Pop.aiff"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def strip_ctml(text: str) -> str:
    text = re.sub(r'</?_[^>]*>', '', text)
    text = re.sub(r'<[^>]+/>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_actions(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'<reachy_mini:(\w+)', text)]


async def main():
    load_env()

    print()
    print("  小白 1.0 — 视觉感知版")
    print("  摄像头检测你的表情，小白根据你的情绪调整回应")
    print("  输入中文聊天，输入 quit 退出")
    print()

    model = build_xiaobai_model()
    safety_gate = SafetyGateChannel()

    # 启动表情追踪
    tracker = EmotionTracker()
    tracker.start()
    print("  ✓ 摄像头表情追踪已启动（后台运行）")
    time.sleep(1)  # 等摄像头初始化

    soul = load_soul()
    style = load_companion_style()
    system_parts = [soul, CTML_INSTRUCTION]
    if style:
        system_parts.append(f"\n## 陪伴风格规则\n\n{style}")

    # 加入视觉感知指令
    system_parts.append("""
## 视觉感知

你能看到陶陶的表情。每轮对话会告诉你他当前的情绪状态。

规则：
- 如果陶陶在笑，你的回应可以更活泼，动作更欢快
- 如果陶陶皱眉或难过，你要更温柔，动作更收敛
- 不要每次都说"我看到你在笑/难过"，自然地根据情绪调整就好
- 偶尔可以轻轻提一下你注意到的表情变化，但不要像监控一样报告
""")

    base_instruction = "\n".join(system_parts)
    agent = Agent(model=model, system_prompt=base_instruction)
    print("  ✓ 小白 Ghost（带视觉）就绪")

    speak("你好呀，我看到你了。")
    print("  小白: 你好呀，我看到你了。")
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

            # 读取当前表情
            vision_context = tracker.get_context()
            print(f"  [👁 {tracker.emotion}]")

            # 安全评估
            evaluation = safety_gate.evaluate_input(child_text)
            redacted = safety_gate.redact(child_text)
            level = evaluation.get("safety_level", "?")
            level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "⚪")

            # 把视觉上下文加到用户消息里
            augmented_input = f"[{vision_context}]\n\n陶陶说：{redacted}"

            try:
                result = await agent.run(
                    user_prompt=augmented_input,
                    message_history=history if history else None,
                )
                ctml_output = result.output
                history.extend(result.new_messages())

                speech = strip_ctml(ctml_output)
                actions = parse_actions(ctml_output)

                if actions:
                    action_sound()
                    print(f"  [{level_icon}] [动作: {', '.join(actions)}]")
                    await asyncio.sleep(0.3)

                print(f"  小白: {speech}")
                speak(speech)
                print()

            except Exception as e:
                print(f"  [错误] {e}\n")

    finally:
        tracker.stop()
        speak("再见啦")
        print("  小白: 再见啦")


if __name__ == "__main__":
    asyncio.run(main())
