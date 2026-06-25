from reachy_mini_conversation_app.xiaobai_contract import (
    MotionStep,
    RuntimeKind,
    make_reply_plan,
    make_official_turn,
)


def test_make_official_turn_uses_official_runtime() -> None:
    """Official turns should carry the official Xiaobai runtime marker."""
    turn = make_official_turn("小白你好")

    assert turn.text == "小白你好"
    assert turn.runtime == RuntimeKind.OFFICIAL_1
    assert turn.source == "official_profile"


def test_make_reply_plan_can_carry_motion_steps() -> None:
    """Reply plans should keep speech and motion together."""
    plan = make_reply_plan(
        "收到",
        motion_steps=[MotionStep(name="move_head", params={"direction": "front"})],
    )

    assert plan.should_reply is True
    assert plan.audio.text == "收到"
    assert plan.motion.sync_to_speech is True
    assert plan.motion.steps[0].name == "move_head"
