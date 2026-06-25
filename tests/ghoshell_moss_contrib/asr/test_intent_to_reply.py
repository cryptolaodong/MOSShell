from ghoshell_moss_contrib.asr.intent_to_reply import decide_intent_to_reply
from ghoshell_moss_contrib.asr.voice_turn_gate import VoiceTurnDecision


def decision(
    text: str,
    *,
    accept: bool,
    reason: str,
    addressed: bool = False,
    active_before: bool = False,
):
    return decide_intent_to_reply(
        text,
        VoiceTurnDecision(
            accept=accept,
            reason=reason,
            addressed=addressed,
            active_before=active_before,
            active_left_seconds=8.0 if active_before else 0.0,
            text_len=len(text),
        ),
    )


def test_unaddressed_background_voice_is_dropped() -> None:
    item = decision("电视里有人在说话", accept=False, reason="idle_not_addressed")

    assert item.action == "drop"
    assert item.background_context
    assert item.reason == "idle_not_addressed_background_context"


def test_unaddressed_typing_noise_is_dropped() -> None:
    item = decision("旁边有打字声音", accept=False, reason="idle_not_addressed")

    assert item.action == "drop"
    assert item.background_context


def test_addressed_noise_instruction_replies() -> None:
    item = decision(
        "小白我旁边有打字声音你只需要回答我听到了",
        accept=True,
        reason="addressed",
        addressed=True,
    )

    assert item.action == "reply"
    assert item.reason == "addressed_direct"
    assert item.background_context
    assert item.direct_request


def test_short_wake_replies() -> None:
    item = decision("小白你好", accept=True, reason="addressed", addressed=True)

    assert item.action == "reply"
    assert item.reason == "addressed_direct"


def test_active_followup_direct_question_replies() -> None:
    item = decision(
        "你现在能做什么",
        accept=True,
        reason="active_followup",
        active_before=True,
    )

    assert item.action == "reply"
    assert item.reason == "active_followup_direct"


def test_active_followup_background_context_drops_without_direct_request() -> None:
    item = decision(
        "电视里有人说话",
        accept=True,
        reason="active_followup",
        active_before=True,
    )

    assert item.action == "drop"
    assert item.reason == "active_followup_background_context"


def test_hold_style_turns_remain_reply_when_gate_rescues_address() -> None:
    item = decision(
        "请等我说完以后再用一句话回答",
        accept=True,
        reason="clipped_address_rescue",
    )

    assert item.action == "reply"
    assert item.reason == "clipped_address_rescue"
