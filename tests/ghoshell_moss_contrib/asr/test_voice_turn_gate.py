from ghoshell_moss_contrib.asr.voice_turn_gate import (
    VoiceTurnGate,
    looks_like_active_followup_request,
    looks_like_clipped_address_request,
    looks_like_prefix_only_greeting,
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_idle_requires_address_word() -> None:
    clock = Clock()
    gate = VoiceTurnGate(clock=clock)

    decision = gate.decide_final("你好")

    assert not decision.accept
    assert decision.reason == "idle_not_addressed"
    assert not decision.addressed


def test_addressed_turn_opens_followup_window() -> None:
    clock = Clock()
    gate = VoiceTurnGate(active_seconds=12, clock=clock)

    first = gate.decide_final("小白你好")
    clock.advance(3)
    second = gate.decide_final("你现在能做什么")

    assert first.accept
    assert first.reason == "addressed"
    assert second.accept
    assert second.reason == "active_followup"
    assert second.active_before


def test_followup_window_expires_back_to_quiet() -> None:
    clock = Clock()
    gate = VoiceTurnGate(active_seconds=12, clock=clock)

    assert gate.decide_final("小白你好").accept
    clock.advance(13)
    decision = gate.decide_final("你现在能做什么")

    assert not decision.accept
    assert decision.reason == "idle_not_addressed"


def test_open_followup_window_allows_recovered_followup() -> None:
    clock = Clock()
    gate = VoiceTurnGate(active_seconds=3, clock=clock)

    gate.open_followup_window()
    clock.advance(1)
    decision = gate.decide_final("现在能做什么")

    assert decision.accept
    assert decision.reason == "active_followup"
    assert decision.active_before


def test_default_idle_partial_does_not_commit_short_background_text() -> None:
    clock = Clock()
    gate = VoiceTurnGate(clock=clock)

    first = gate.decide_partial("你好")
    clock.advance(5)
    second = gate.decide_partial("你好你好你好")

    assert not first.commit
    assert first.reason == "idle_unaddressed_wait"
    assert not second.commit


def test_idle_partial_commit_is_explicitly_configured() -> None:
    clock = Clock()
    gate = VoiceTurnGate(idle_partial_chars=4, idle_partial_seconds=1.0, clock=clock)

    assert not gate.decide_partial("背景声音测试").commit
    clock.advance(1.1)
    decision = gate.decide_partial("背景声音测试")

    assert decision.commit
    assert decision.reason == "idle_unaddressed_commit"


def test_clipped_address_request_rescue_matches_explicit_requests() -> None:
    prefixes = ("我想", "请你", "请用", "你能", "来测试", "不要")
    keywords = ("测试", "回答", "做什么", "说完", "最后")

    assert looks_like_clipped_address_request(
        "我想测试一下长句子的理解和延迟，请你用一句话回答我",
        prefixes=prefixes,
        keywords=keywords,
    )
    assert looks_like_clipped_address_request(
        "请你用一句话回答我现在能做什么",
        prefixes=prefixes,
        keywords=keywords,
    )
    assert looks_like_clipped_address_request(
        "請用聽話解單回答你今天最喜歡什麼字為什麼",
        prefixes=prefixes,
        keywords=keywords,
    )
    assert looks_like_clipped_address_request(
        "来测试你会不会抢答，请等我把这句话全部说完以后，再用一句话回答我听明白了。",
        prefixes=prefixes,
        keywords=keywords,
    )
    assert looks_like_clipped_address_request(
        "不要在中间停止时候唱话最后持续要说我听你来了",
        prefixes=prefixes,
        keywords=keywords,
    )
    assert not looks_like_clipped_address_request(
        "你好",
        prefixes=prefixes,
        keywords=keywords,
    )
    assert not looks_like_clipped_address_request(
        "背景里有人聊天但是没有明确请求",
        prefixes=prefixes,
        keywords=keywords,
    )


def test_active_followup_request_filter_rejects_short_noise() -> None:
    assert looks_like_active_followup_request("你现在能做什么")
    assert looks_like_active_followup_request("继续")
    assert looks_like_active_followup_request("请再说一遍")
    assert looks_like_active_followup_request("动动脑袋")

    assert not looks_like_active_followup_request("感谢你长得")
    assert not looks_like_active_followup_request("水水水水水")
    assert not looks_like_active_followup_request("你好")


def test_prefix_only_greeting_matches_short_wake_phrases_only() -> None:
    assert looks_like_prefix_only_greeting("小白")
    assert looks_like_prefix_only_greeting("小白你好")
    assert looks_like_prefix_only_greeting("你好，小白")
    assert looks_like_prefix_only_greeting("小白在吗")
    assert looks_like_prefix_only_greeting("小白你在吗")

    assert not looks_like_prefix_only_greeting("你好")
    assert not looks_like_prefix_only_greeting("小白你现在能做什么")
    assert not looks_like_prefix_only_greeting("小白请你动动脑袋")
    assert not looks_like_prefix_only_greeting("小白你好你现在能做什么")
