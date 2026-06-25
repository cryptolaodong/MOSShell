# 小白 1.0 — 回合制 Reachy Mini 伴侣

这是小白的独立 1.0 研究线：先回到稳定、可验收的回合制体验，不继续把主风险压在 always-on 语音、VAD、ASR、回声和 MOSS 调度上。

## 为什么回到 1.0

小白 2.0 的实时语音目标仍然成立，但当前链路同时耦合了麦克风、VAD、ASR、TTS、动作同步、机器人 daemon 和 MOSS mindflow，现场调试成本太高。小白 1.0 的目标是先把“能稳定对话、能同步动作、能解释问题、能逐步扩展”做成可靠底座。

## 当前状态

- [x] 回合制文字输入
- [x] 安全策略对接
- [x] Qwen/DashScope 生成回复
- [x] CTML 风格动作标签
- [x] Reachy Mini `goto_target` 安全动作
- [x] Mac TTS 语音兜底
- [x] 多轮上下文
- [x] 面部追踪代码保留为后续能力

## 一条命令验证

```bash
cd /Users/laodong/Documents/xiaobai-moss
/Users/laodong/Documents/MOSShell/.venv/bin/python run_xiaobai_real.py
```

详细验证步骤见 [VERIFY.md](VERIFY.md)。

## 架构

```text
xiaobai-moss/
├── run_xiaobai_real.py      # 真实硬件主入口
├── VERIFY.md                # 现场验收指南
├── channels/
│   ├── safety_gate.py       # 安全策略 Channel
│   ├── reachy_mini_sim.py   # 模拟 Channel
│   └── face_tracker.py      # 面部追踪能力
├── ghost/
│   ├── soul.md              # 小白人格
│   └── xiaobai_ghost.py     # 模型构建
├── data/reachy_mini_emotions/
│   └── *.json, *.wav        # Reachy Mini 表情/动作资源
├── docs/
│   └── GHOSTINSHELLS_REUSE_NOTES.md
├── .env.example             # 本地配置模板
└── pyproject.toml
```

依赖的本地项目：

```text
../Reachy/                   # 安全策略、工具脚本、现场文档
../MOSShell/                 # MOSS 框架和部分 Reachy 集成代码
```

## 已验证的主链路

```text
用户打字输入
    -> 安全策略评估
    -> LLM 生成带动作标签的回复
    -> 提取动作 + 纯文本
    -> Reachy Mini 做同步动作
    -> Mac/配置的 TTS 说话
    -> 等待下一轮输入
```

## 小白 1.0 验收标准

1. 连续 20 轮文字输入，至少 19 轮有回复。
2. 不会没输入时自己回复。
3. 不会自己打断自己。
4. 简单动作请求能执行，并且动作与语音大致同步。
5. 普通问题 3-5 秒内开始输出。
6. 出错时能看出卡在连接、LLM、TTS 还是动作。

## GhostInShells/moss-in-reachy-mini 的使用策略

该上游项目当前声明 Apache License 2.0，可以借鉴和复用，但小白 1.0 不做整包替换。优先研究并吸收：

- `AudioMixer` / `sound` / `file_player`
- `MicHub`
- 回合制或 PTT 状态机
- 音乐/音效播放能力

暂不照搬：

- always-on 实时语音主链路
- 大规模 app/直播/视觉/记忆能力
- 依赖很重的 GStreamer/WebRTC 流式播放链路

详见 [docs/GHOSTINSHELLS_REUSE_NOTES.md](docs/GHOSTINSHELLS_REUSE_NOTES.md)。

## 待优化

- 把 Mac TTS 替换为更稳定的可配置 TTS。
- 将语音输入做成明确按钮/快捷键触发的回合制 STT。
- 将网页控制台接入启动/停止/状态查看。
- 将动作和语音同步封装成最小稳定模块。
- 把 Reachy 内置麦克风作为后续能力，不作为 1.0 阻塞项。

## 本地配置

复制模板后填入本机密钥：

```bash
cp .env.example .moss_ws/.env
```

不要把 `.moss_ws/.env` 提交到 Git。
