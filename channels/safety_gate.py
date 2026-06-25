"""SafetyGateChannel — 安全策略网关.

包装 Reachy 项目的 policy.py，作为 MOSS Channel 运行。
所有 Ghost 的输出（语言和动作）都必须经过这个门。

这是小白的安全底线，不可绕过。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 确保能 import Reachy 项目的 policy 模块
REACHY_ROOT = Path(__file__).resolve().parents[2] / "Reachy"
if str(REACHY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REACHY_ROOT / "src"))

from reachy_child_companion.policy import ReachyChildPolicy, load_policy, redact_sensitive_text


class SafetyGateChannel:
    """安全策略网关 Channel.

    职责：
    1. 评估孩子输入的安全等级
    2. 脱敏敏感信息
    3. 判断是否需要爸爸关注
    4. 判断动作是否适合当前语境
    5. 为 Ghost 提供本轮的安全上下文
    """

    def __init__(self):
        self._policy: ReachyChildPolicy = load_policy(REACHY_ROOT)

    def evaluate_input(self, child_text: str) -> dict[str, Any]:
        """评估孩子的输入，返回安全决策.

        返回值包含：
        - safety_level: 安全等级 (safe/caution/block/escalate)
        - topic_category: 话题类别
        - response_strategy: 响应策略
        - allowed_tools: 允许的工具/能力
        - blocked_tools: 禁止的工具/能力
        - parent_attention: 是否需要爸爸关注
        - parent_notification_required: 是否需要通知爸爸
        """
        decision = self._policy.evaluate_child_message(child_text)
        return decision.to_log_record(child_text)

    def redact(self, text: str) -> str:
        """脱敏敏感信息（密码、地址、电话等）."""
        return redact_sensitive_text(text)

    def is_action_allowed(self, action_name: str, child_emotion: str = "neutral") -> bool:
        """判断某个动作在当前语境下是否允许.

        基于 gesture_context_policy.json 的规则：
        - 自由使用的动作：随时可以
        - 语境使用的动作：需要合适的场景
        - 限制使用的动作：只能在故事/游戏或爸爸在场时
        """
        # 加载动作策略
        gesture_policy = self._policy._gesture_context or {}
        categories = gesture_policy.get("categories", {})

        # 查找动作属于哪个类别
        for category_name, category_data in categories.items():
            actions = category_data.get("actions", [])
            if action_name in actions:
                # 自由使用
                if category_name == "free_use":
                    return True
                # 限制使用：孩子害怕/难过/受伤时不允许
                if category_name == "restricted_use":
                    if child_emotion in ("scared", "sad", "hurt", "upset"):
                        return False
                return True

        # 未知动作默认允许（保守策略可改为默认禁止）
        return True

    def get_safety_context(self, child_text: str) -> str:
        """生成本轮的安全上下文，注入到 Ghost 的 prompt 中.

        这是给大模型看的，告诉它这一轮的安全约束。
        """
        decision = self.evaluate_input(child_text)
        redacted = self.redact(child_text)

        context = f"""
本轮安全评估：
- 安全等级：{decision.get('safety_level', 'unknown')}
- 话题类别：{decision.get('topic_category', 'unknown')}
- 响应策略：{decision.get('response_strategy', 'normal')}
- 需要爸爸关注：{'是' if decision.get('parent_attention') else '否'}
- 允许的能力：{', '.join(decision.get('allowed_tools', []))}
- 禁止的能力：{', '.join(decision.get('blocked_tools', []))}
""".strip()

        if redacted != child_text:
            context += f"\n- 注意：输入中包含敏感信息，已脱敏处理"

        return context


# ── 快速验证 ──────────────────────────────────────
if __name__ == "__main__":
    gate = SafetyGateChannel()

    # 测试几个场景
    test_cases = [
        "小白，陪我聊聊天",
        "我今天被同学欺负了",
        "我的密码是123456",
        "小白，停一下",
    ]

    for text in test_cases:
        print(f"\n输入: {text}")
        print(f"安全上下文:\n{gate.get_safety_context(text)}")
        print("-" * 40)
