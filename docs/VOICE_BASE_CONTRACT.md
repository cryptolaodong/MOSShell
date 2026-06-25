# 小白语音底座与功能插座约定

目标：后续能力不要只兼容小白 1.0，也不要只兼容小白 2.0。语音、动作、人格、记忆、工具、网页控制台都应该通过一层稳定 contract 接入。

## 核心原则

1. 功能模块不直接依赖 MOSS realtime，也不直接依赖 1.0 的命令行循环。
2. 1.0 和 2.0 各自只负责把自己的输入/输出转换成共享结构。
3. 能力模块只处理“已经确认的一轮用户输入”，并返回“该说什么、做什么动作、是否应该回复”。
4. 麦克风、VAD、ASR、回声抑制、按键录音、网页按钮都属于 input adapter，不泄漏到业务功能里。
5. TTS、机器人喇叭、Mac 喇叭、动作同步都属于 output adapter，不泄漏到业务功能里。

## 共享数据流

```text
TurnInputAdapter
    -> TurnInput
    -> BrainAdapter / Feature Module
    -> ReplyPlan
    -> AudioOutputAdapter + MotionOutputAdapter
```

## 小白 1.0 适配

```text
文字输入 / 按键录音
    -> TurnInput(runtime=xiaobai_1_turn_based)
    -> 安全策略 + LLM/规则功能
    -> ReplyPlan
    -> Mac TTS / Reachy goto_target
```

1.0 优先保证稳定、可手动验收、可调试。

## 小白 2.0 适配

```text
MOSS voice event / realtime ASR
    -> TurnInput(runtime=xiaobai_2_moss_realtime)
    -> 同一套安全策略 + LLM/规则功能
    -> ReplyPlan
    -> upload TTS / speech sync / body command
```

2.0 继续研究实时体验，但不能要求功能模块知道 MOSS 内部事件细节。

## 新功能开发要求

新能力必须回答三个问题：

1. 输入是否只依赖 `TurnInput`？
2. 输出是否只返回 `ReplyPlan`？
3. 是否同时能被 1.0 adapter 和 2.0 adapter 调用？

如果答案不是三个“是”，先补 adapter 或 contract，不要把功能绑死在某一条链路里。

## 当前代码入口

共享 contract 先放在：

```text
xiaobai_core/contracts.py
```

后续 1.0 和 2.0 都围绕这层做适配，不再让具体功能直接调用麦克风、ASR、TTS 或 MOSS mindflow。
