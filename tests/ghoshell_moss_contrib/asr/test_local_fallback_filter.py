from ghoshell_moss_contrib.asr.async_states import (
    _combine_open_local_fallback_fragments,
    _canonicalize_open_local_fallback_text,
    _canonicalize_safe_local_fallback_text,
    _is_safe_open_local_fallback_text,
    _is_safe_local_fallback_text,
    _looks_like_high_rms_ability_request_mishear,
    _looks_like_high_rms_short_wake_mishear,
    _looks_like_open_local_fallback_prefix_fragment,
    _looks_like_open_request_fragment,
    _looks_like_rescuable_wake_second_pass,
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
    assert _canonicalize_safe_local_fallback_text("小白以後") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("走來你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白腰") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白魚") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白衣") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("我把你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("我拜你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小泥好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("來你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("想玩你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("做完你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小白天啊") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("希望拜你好") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小蛋糕") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("小蛋一跑") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("早白一趟") == "小白你好"
    assert _canonicalize_safe_local_fallback_text("二个小白你好") == "小白你好"


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
    assert not _is_safe_local_fallback_text("我管你好")
    assert not _is_safe_local_fallback_text("下个月了")
    assert not _is_safe_local_fallback_text("你好")
    assert not _is_safe_local_fallback_text("谢谢你")
    assert not _is_safe_local_fallback_text("小白你好小白你好")
    assert not _is_safe_local_fallback_text("小白你好小白你好小白你好小白你好")
    assert not _is_safe_local_fallback_text("你不在这儿怎么办一件事")
    assert not _is_safe_local_fallback_text("小白你好 我想")


def test_rescue_prompt_only_allows_wake_like_fragments() -> None:
    for text in ("小白癢", "小一號", "小孩好", "叫白魚好", "小白吧"):
        assert _looks_like_rescuable_short_wake_fragment(_normalize_local_asr_text(text))
    for text in ("就好", "小鈴好", "小明好", "小白米酒", "小白米", "找一号", "我"):
        assert not _looks_like_rescuable_short_wake_fragment(_normalize_local_asr_text(text))


def test_rescue_model_only_runs_on_short_wake_like_fragments() -> None:
    for text in ("小丸你好", "小玩意好", "想办你好", "叫白你好", "小雷"):
        assert _looks_like_rescuable_wake_second_pass(_normalize_local_asr_text(text))
    for text in ("我", "請問一下", "小明好", "小白米酒", "背景声音测试"):
        assert not _looks_like_rescuable_wake_second_pass(_normalize_local_asr_text(text))


def test_high_rms_short_wake_mishear_is_narrow() -> None:
    for text in ("等待米好", "小白米", "小白猫", "早掰你好"):
        assert _looks_like_high_rms_short_wake_mishear(_normalize_local_asr_text(text))

    for text in ("小白米酒", "小明好", "背景声音测试", "等待我说完", "小白你好我想"):
        assert not _looks_like_high_rms_short_wake_mishear(_normalize_local_asr_text(text))

    assert not _is_safe_local_fallback_text("小白米")


def test_high_rms_ability_request_mishear_is_narrow() -> None:
    assert _looks_like_high_rms_ability_request_mishear(_normalize_local_asr_text("你進化回啦"))

    for text in ("电视里有人问现在能做什么", "你进化了吗", "现在能做什么"):
        assert not _looks_like_high_rms_ability_request_mishear(_normalize_local_asr_text(text))


def test_open_local_fallback_accepts_addressed_questions() -> None:
    assert _is_safe_open_local_fallback_text("小白晴,用一句话简单回答你,今天最喜欢什么颜色,为什么?")
    assert (
        _canonicalize_open_local_fallback_text(
            "小白晴,用一句话简单回答你,今天最喜欢什么颜色,为什么?"
        )
        == "小白请用一句话简单回答你今天最喜欢什么颜色为什么"
    )
    assert _is_safe_open_local_fallback_text("小孩请用一句话解的回答你今天最喜欢什么演奏为什么")
    assert _canonicalize_open_local_fallback_text("想白情簡短回答你最喜歡做什麼") == (
        "小白请简短回答你最喜欢做什么"
    )
    assert _is_safe_open_local_fallback_text("想把你覺得北京這個城市能量,請用一句話回答")
    assert (
        _canonicalize_open_local_fallback_text("想把你覺得北京這個城市能量,請用一句話回答")
        == "小白你觉得北京这个城市能量请用一句话回答"
    )
    assert _is_safe_open_local_fallback_text("想把你覺得北京這個城市怎麼樣?解釋你一句話回答")
    assert _is_safe_open_local_fallback_text("想完你覺得北京這個城市怎麼樣?請用一句話回答")
    assert _is_safe_open_local_fallback_text("小怪你好你现在能做什么")
    assert (
        _canonicalize_open_local_fallback_text("小怪你好你现在能做什么")
        == "小白你好你现在能做什么"
    )
    assert _is_safe_open_local_fallback_text("嗯…小白你好你现在能做什么")
    assert (
        _canonicalize_open_local_fallback_text("嗯…小白你好你现在能做什么")
        == "小白你好你现在能做什么"
    )
    assert _is_safe_open_local_fallback_text(
        "小白你好請的摸把這句話全部說完以後,再一一句話回答"
    )
    assert _is_safe_open_local_fallback_text(
        "想白您好請的摸把這句話全部說完以後,再一一句話回答"
    )
    assert _canonicalize_open_local_fallback_text(
        "請用聽話解單回答你今天最喜歡什麼字為什麼"
    ).startswith("请用")
    assert _is_safe_open_local_fallback_text("今天最喜欢什么颜色会什么")
    clipped_turn = "不要在中间停止时候唱话最后持续要说我听你来了"
    assert _is_safe_open_local_fallback_text(clipped_turn)
    assert _canonicalize_open_local_fallback_text(clipped_turn).startswith("小白不要在中间")
    assert _is_safe_open_local_fallback_text(
        "来测试你会不会抢答，请等我把这句话全部说完以后，再用一句话回答我听明白了。"
    )
    assert _is_safe_open_local_fallback_text(
        "此來測試你會不會強大?新的火把這句話全部說完以後再用一句話回答我聽明白了"
    )
    assert _is_safe_open_local_fallback_text(
        "小白請不要在中間停頓的時候叉划,最後還需要說我聽明白了"
    )
    assert _is_safe_open_local_fallback_text(
        "小白請不要在中間評論的時候巧達最後需要說我聽你問了"
    )
    assert _is_safe_open_local_fallback_text(
        "小白請不要在中間停頓的時候搶答即使要說我聽你明白了"
    )
    assert _is_safe_open_local_fallback_text("最後参划,最後只需要說我清明白了")
    assert _is_safe_open_local_fallback_text("不要搶答等我結束之後再回")
    assert _is_safe_open_local_fallback_text(
        "小白一好 我想測試 讓據會不會被李中篤打斷請最後再說一句話"
    )
    assert _is_safe_open_local_fallback_text(
        "小白你好,这是你条长俊测试,请不用强达,等我结束之后再回复"
    )
    assert _is_safe_open_local_fallback_text(
        "小白你好,我旁邊有打字聲音,你只需要回擋我身上了"
    )
    assert _is_safe_open_local_fallback_text(
        "小白你好,如果便是理有人说话,你应该的国家你之后再回答"
    )
    assert _is_safe_open_local_fallback_text("小白你好如果聽到欠牌聲你不要接話")
    assert _is_safe_open_local_fallback_text("小白你好旁边有视频生意你听到我叫小白再回答")
    assert _is_safe_open_local_fallback_text("找你航米現代能做什麼請用一句話回答")


def test_open_local_fallback_stores_addressed_test_prefix_without_replying() -> None:
    prefix = "小白你好我想測試"
    assert _looks_like_open_local_fallback_prefix_fragment(prefix)
    assert not _is_safe_open_local_fallback_text(prefix)

    combined = _combine_open_local_fallback_fragments(prefix, "不要搶答等我結束之後再回")
    assert combined.startswith("小白你好我想测试")
    assert _is_safe_open_local_fallback_text(combined)


def test_open_local_fallback_combines_addressed_answer_prefix_with_semantic_suffix() -> None:
    prefix = "小白你好情依句话回答你"
    suffix = "不喜欢什么颜色为什么"
    assert _looks_like_open_local_fallback_prefix_fragment(prefix)
    assert not _is_safe_open_local_fallback_text(prefix)
    assert not _is_safe_open_local_fallback_text(suffix)

    combined = _combine_open_local_fallback_fragments(prefix, suffix)
    assert combined.startswith("小白你好请一句话回答你")
    assert _is_safe_open_local_fallback_text(combined)


def test_open_local_fallback_combines_split_turn_completion_fragments() -> None:
    first = "小板請不要在中間停頓的時候叉划,最後"
    second = "只需要说我听明白了"
    assert _looks_like_open_local_fallback_prefix_fragment(first)
    assert not _is_safe_open_local_fallback_text(first)
    assert not _is_safe_open_local_fallback_text(second)
    combined = _combine_open_local_fallback_fragments(first, second)
    assert combined.startswith("小白请不要在中间")
    assert _is_safe_open_local_fallback_text(combined)

    noisy_first = "小白請不要在中間停滾的時候差跨,最後"
    noisy_second = "只需要收我心你败了"
    assert _looks_like_open_local_fallback_prefix_fragment(noisy_first)
    assert not _is_safe_open_local_fallback_text(noisy_first)
    noisy_combined = _combine_open_local_fallback_fragments(noisy_first, noisy_second)
    assert noisy_combined.startswith("小白请不要在中间")
    assert _is_safe_open_local_fallback_text(noisy_combined)

    clipped_first = "要說一個長句子來自使你會補會強大,請等我把這句話全部收完以後"
    clipped_second = "再用一句话回答我新明白了"
    assert _looks_like_open_local_fallback_prefix_fragment(clipped_first)
    assert not _is_safe_open_local_fallback_text(clipped_first)
    clipped_combined = _combine_open_local_fallback_fragments(clipped_first, clipped_second)
    assert clipped_combined.startswith("小白要说")
    assert _is_safe_open_local_fallback_text(clipped_combined)


def test_open_local_fallback_prefix_does_not_commit_as_short_wake() -> None:
    prefix = "小白你好請等我把這句話全部輸"
    assert _looks_like_open_local_fallback_prefix_fragment(prefix)
    assert not _is_safe_open_local_fallback_text(prefix)
    assert not _is_safe_local_fallback_text(prefix)


def test_open_local_fallback_rejects_short_greeting_and_noise() -> None:
    assert not _is_safe_open_local_fallback_text("小白你好")
    assert not _is_safe_open_local_fallback_text("小怪你好")
    assert not _is_safe_open_local_fallback_text("背景声音测试")
    assert not _is_safe_open_local_fallback_text("办公室打字声音测试")
    assert not _is_safe_open_local_fallback_text("电视里面有人说今天最喜欢什么颜色为什么")
    assert not _is_safe_open_local_fallback_text("如果电视里有人说话你应该之后再回答")
    assert not _is_safe_open_local_fallback_text("旁边同事聊天说请用一句话回答")
    assert not _is_safe_open_local_fallback_text("这是你条长俊测试请不用强达等我结束之后再回复")
    assert not _is_safe_open_local_fallback_text("小白请简单回答机器人")
    assert _looks_like_open_request_fragment("")
    assert _looks_like_open_request_fragment("请用一句话")
    assert not _looks_like_open_request_fragment("背景声音测试")
    assert not _looks_like_open_request_fragment("办公室打字声音测试")
