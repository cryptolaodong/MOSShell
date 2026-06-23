from ghoshell_moss_contrib.asr.voice_turn_gate import VoiceTurnGate


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
