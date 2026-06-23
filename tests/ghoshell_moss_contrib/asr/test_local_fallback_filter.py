from ghoshell_moss_contrib.asr.async_states import (
    _canonicalize_safe_local_fallback_text,
    _is_safe_local_fallback_text,
    _looks_like_rescuable_short_wake_fragment,
    _normalize_local_asr_text,
)


def test_safe_local_fallback_accepts_short_entry_phrases() -> None:
    assert _is_safe_local_fallback_text("小白你好")
    assert _is_safe_local_fallback_text("想掰你好")
    assert _is_safe_local_fallback_text("你現在能做什麼")


def test_safe_local_fallback_canonicalizes_wake_homophone() -> None:
    assert _canonicalize_safe_local_fallback_text("想掰你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("想法你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小班你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小番茗好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白米好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白糖") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白一號") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("老白你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白魚好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("叫白米好") == "小白你好"


def test_safe_local_fallback_canonicalizes_noisy_ability_question() -> None:
    assert _is_safe_local_fallback_text("小白底線才能做")
    assert _canonicalize_safe_local_fallback_text("小白底線才能做") == "小白你现在能做什么"
    assert _is_safe_local_fallback_text("想把你現在能夠什麼")
    assert _canonicalize_safe_local_fallback_text("想把你現在能夠什麼") == "小白你现在能做什么"


def test_safe_local_fallback_rejects_likely_fragments() -> None:
    assert not _is_safe_local_fallback_text("嗯")
    assert not _is_safe_local_fallback_text("就好")
    assert not _is_safe_local_fallback_text("小鈴好")
    assert not _is_safe_local_fallback_text("小明好")
    assert not _is_safe_local_fallback_text("小白米酒")
    assert not _is_safe_local_fallback_text("要重視在外面")
    assert not _is_safe_local_fallback_text("小白米")
    assert not _is_safe_local_fallback_text("找一号")
    assert not _is_safe_local_fallback_text("我")


def test_rescue_prompt_only_allows_wake_like_fragments() -> None:
    for text in ("小白癢", "小一號", "小孩好", "叫白魚好", "小白吧"):
        assert _looks_like_rescuable_short_wake_fragment(_normalize_local_asr_text(text))
    for text in ("就好", "小鈴好", "小明好", "小白米酒", "小白米", "找一号", "我"):
        assert not _looks_like_rescuable_short_wake_fragment(_normalize_local_asr_text(text))
