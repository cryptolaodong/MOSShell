from ghoshell_moss_contrib.asr.vad import EnergyVAD


def test_energy_vad_default_hold_is_voice_latency_tuned(monkeypatch) -> None:
    monkeypatch.delenv("MOSS_ASR_ENERGY_SILENCE_HOLD_SECONDS", raising=False)

    vad = EnergyVAD()

    assert vad._silence_hold_time == 0.65


def test_energy_vad_hold_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("MOSS_ASR_ENERGY_SILENCE_HOLD_SECONDS", "0.9")

    vad = EnergyVAD()

    assert vad._silence_hold_time == 0.9
