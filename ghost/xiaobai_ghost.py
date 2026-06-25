"""小白 Ghost 定义.

基于 MOSS 的 AtomMeta，用 DashScope (Qwen) 作为大脑，
引用现有项目的安全策略和陪伴人格作为灵魂。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# 把 Reachy 项目加入 path，这样可以 import policy 模块
REACHY_ROOT = Path(__file__).resolve().parents[2] / "Reachy"
if str(REACHY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REACHY_ROOT / "src"))

XIAOBAI_MOSS_ROOT = Path(__file__).resolve().parents[1]


def load_env():
    """加载 .moss_ws/.env 环境变量."""
    env_path = XIAOBAI_MOSS_ROOT / ".moss_ws" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def build_xiaobai_model() -> OpenAIModel:
    """构建 DashScope 兼容的 OpenAI 模型实例."""
    load_env()

    base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    model_name = os.environ.get("XIAOBAI_MODEL", "qwen-plus")

    if not api_key:
        raise RuntimeError(
            "需要设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量。"
            "请编辑 .moss_ws/.env 文件。"
        )

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(model_name=model_name, provider=provider)


def load_soul() -> str:
    """加载小白的灵魂文件."""
    soul_path = Path(__file__).parent / "soul.md"
    return soul_path.read_text(encoding="utf-8")


def load_companion_style() -> str:
    """从 Reachy 项目加载陪伴风格."""
    style_path = REACHY_ROOT / "config" / "companion_style.md"
    if style_path.exists():
        return style_path.read_text(encoding="utf-8")
    return ""


def load_growth_coach() -> str:
    """从 Reachy 项目加载成长教练 prompt."""
    coach_path = REACHY_ROOT / "config" / "growth_coach_prompt.md"
    if coach_path.exists():
        return coach_path.read_text(encoding="utf-8")
    return ""
