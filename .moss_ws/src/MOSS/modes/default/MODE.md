---
# Mode 元数据（YAML frontmatter）。
# 以下字段由 moss modes create 自动填充，也可以手动编辑。
apps:
  - 'bodies/reachymini'  # 语音机器人常驻模式只暴露必要 app，减少每轮 LLM 上下文。
  - 'sensors/voice'
bringup_apps:       # 启动时自动 bringup 的 app 列表
  - 'bodies/reachymini'
  - 'sensors/voice'
ctml_version: 'reachy_fast'  # Reachy 语音模式使用精简 CTML，降低每轮 LLM 首 token 延迟
description: ''     # 一行描述
name: 'default'     # mode 名称（与目录名一致）
---

在此写 mode 的使用说明，支持 markdown。
会显示在 moss modes show 的输出中。
对于 AI 模型，这里的文字会作为 mode instruction 注入上下文。
