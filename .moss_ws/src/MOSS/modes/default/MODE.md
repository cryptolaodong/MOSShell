---
# Mode 元数据（YAML frontmatter）。
# 以下字段由 moss modes create 自动填充，也可以手动编辑。
apps:
  - '*/*'           # 允许的 app 白名单。'*/*' = 全部公开 app
                    #   group/* = 某个 group 下所有 app
                    #   _ 前缀（如 _private/app）= 禁止访问
bringup_apps:       # 启动时自动 bringup 的 app 列表
  - 'bodies/reachymini'
  - 'sensors/voice'
ctml_version: ''    # 留空使用默认 CTML 版本
description: ''     # 一行描述
name: 'default'     # mode 名称（与目录名一致）
---

在此写 mode 的使用说明，支持 markdown。
会显示在 moss modes show 的输出中。
对于 AI 模型，这里的文字会作为 mode instruction 注入上下文。
