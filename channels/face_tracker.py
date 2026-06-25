"""面部追踪 Channel — 用 MediaPipe FaceLandmarker (新版 API).

用 Mac 摄像头实时检测面部表情，输出情绪分类。
你会看到摄像头窗口 + 情绪标签，用来验证视觉感知能力。
"""
from __future__ import annotations

import time
import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")


def classify_emotion(face_landmarks) -> tuple[str, float]:
    """根据面部关键点判断情绪.

    使用简单的几何特征：
    - 嘴角上扬 → happy
    - 嘴角下拉 → sad
    - 眉毛上挑 + 嘴张开 → surprised
    - 其他 → neutral
    """
    if not face_landmarks:
        return "no_face", 0.0

    landmarks = face_landmarks[0]  # 第一张脸

    try:
        # 嘴巴张开程度
        mouth_top = landmarks[13]  # 上唇中
        mouth_bottom = landmarks[14]  # 下唇中
        mouth_open = abs(mouth_top.y - mouth_bottom.y)

        # 脸高度（归一化用）
        forehead = landmarks[10]
        chin = landmarks[152]
        face_height = abs(forehead.y - chin.y)
        if face_height < 0.01:
            return "neutral", 0.5

        mouth_ratio = mouth_open / face_height

        # 嘴角高度
        mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        avg_corner_y = (left_corner.y + right_corner.y) / 2
        smile_score = (mouth_center_y - avg_corner_y) / face_height

        # 眉毛
        left_brow = landmarks[66]
        left_eye = landmarks[159]
        brow_raise = (left_eye.y - left_brow.y) / face_height

        # 分类
        if smile_score > 0.02:
            return "happy", min(smile_score * 20, 1.0)
        elif smile_score < -0.015:
            return "sad", min(abs(smile_score) * 20, 1.0)
        elif mouth_ratio > 0.08:
            return "surprised", min(mouth_ratio * 5, 1.0)
        elif brow_raise < 0.04:
            return "frustrated", 0.5
        else:
            return "neutral", 0.6

    except (IndexError, AttributeError):
        return "neutral", 0.0


def run_face_tracker_demo():
    """运行面部追踪 demo.

    弹出摄像头窗口，实时检测表情并显示情绪标签。
    按 'q' 退出。
    """
    # 创建 FaceLandmarker
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print("面部追踪已启动。按 'q' 退出。")
    print("对着摄像头做表情：笑、皱眉、张嘴惊讶、面无表情...")
    print()

    emoji_map = {
        "happy": "happy :)",
        "sad": "sad :(",
        "surprised": "surprised :O",
        "frustrated": "frustrated :/",
        "neutral": "neutral :|",
        "no_face": "no face",
    }

    last_print_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # 转换为 MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 检测
        result = landmarker.detect(mp_image)

        emotion = "no_face"
        confidence = 0.0

        if result.face_landmarks:
            emotion, confidence = classify_emotion(result.face_landmarks)

            # 画一些关键点
            for landmark in result.face_landmarks[0][:100]:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # 显示情绪
        label = f"{emoji_map.get(emotion, '?')} ({confidence:.0%})"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 每秒打印一次
        now = time.time()
        if now - last_print_time > 1.0:
            print(f"  {emoji_map.get(emotion, '?')} (confidence: {confidence:.0%})")
            last_print_time = now

        cv2.imshow("XiaoBai Face Tracker", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\n面部追踪已停止。")


if __name__ == "__main__":
    run_face_tracker_demo()
