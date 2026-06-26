# Xiaobai App Pack R1-D 官方 App 启动 Runbook

时间：2026-06-26 Asia/Shanghai

结果：应用层 launcher 已落地，不修改官方 Conversation App 核心 runtime。

启动：

```bash
scripts/start_xiaobai_official_app.sh
```

dry-run：

```bash
scripts/start_xiaobai_official_app.sh --dry-run --skip-output-check
```

launcher 做的事：

- 停止旧 `moss-ghost` screen，避免旧 MOSS runtime 抢 media/action。
- 启动 Memory Candidate sidecar。
- 设置 `xiaobai_app_pack_r1` external profile 环境。
- 启动 `uv run reachy-mini-conversation-app --no-camera --ui`。
- 检查 `/ready`、`/status` 和 Reachy 输出健康。

失败时看返回 JSON 的 `next_steps`。日志默认在 `~/.local/state/xiaobai_app_pack/`。

验收：

```text
tests/test_xiaobai_official_app_launcher.py: 6 passed
tests/: 260 passed
真实启动: /ready=true, backend_connected=true, Reachy output check ok
```
