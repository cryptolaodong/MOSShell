# GhostInShells/moss-in-reachy-mini 复用笔记

检查时间：2026-06-25

上游地址：https://github.com/GhostInShells/moss-in-reachy-mini

## 许可证判断

上游 `pyproject.toml` 当前声明：

```toml
license = { text = "Apache License 2.0" }
```

因此可以在遵守 Apache 2.0 条款的前提下参考、复用和改造代码。若后续直接复制文件，应保留原始版权/许可证说明，并在本项目中记录来源。

## 是否适合直接照搬

不建议整包照搬。

原因：

- 上游是完整 MOSS/Reachy Mini 应用结构，依赖较多。
- 它更偏实时、多媒体、多 app 能力；小白 1.0 的核心目标是回合制稳定。
- 整包替换会再次把麦克风、WebRTC、GStreamer、ASR、TTS、动作和状态机耦合在一起。

## 适合优先吸收的模块

1. `audio/mixer.py`

   用于统一管理 TTS、提示音、音乐和音效的输出层。适合小白 1.0 做“音频中控”。

2. `audio/file_player.py`

   用于本地音频文件播放。适合做提示音、故事音频、测试音和动作音效。

3. `components/sound.py`

   可借鉴它把声音能力封装成组件的方式，但小白 1.0 应保持 API 更简单。

4. `audio/mic_hub.py`

   适合后续回合制 STT、诊断探针、录音测试共享同一个麦克风流，避免多进程抢麦。

5. `framework/listener/*`

   可以借鉴 PTT/commit_reason/状态机，但只用于“按键开始、明确结束”的回合制输入，不直接开启 always-on。

## 小白 1.0 推荐路线

```text
按钮/快捷键开始
    -> 录音
    -> 明确停止或短静音结束
    -> STT
    -> 安全策略
    -> LLM
    -> TTS + 动作同步
    -> 回到等待
```

这个路线把“是否该听、是否该回复”的问题交给明确交互，而不是让系统一直猜。

## 不进入 1.0 的能力

- always-on 自然唤醒
- 声纹识别
- 长期记忆
- 摄像头持续感知
- 直播/音乐 DJ
- 自动主动插话
- 多 app 自动路由和执行

这些能力可作为 2.0 或 1.x 后续实验，不作为小白 1.0 验收阻塞项。
