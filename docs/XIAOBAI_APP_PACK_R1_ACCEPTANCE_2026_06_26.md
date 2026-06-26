# Xiaobai App Pack R1 实机接入验收

时间：2026-06-26 12:20-12:22 Asia/Shanghai

## 目标

在 Pollen 官方 `reachy_mini_conversation_app` 上加载小白外部 profile，验证小白人格、工具白名单、安全动作、声音输出、回滚路径，并确认不修改官方核心运行时。

## 结果

通过。

## 关键证据

官方 app 使用以下环境启动：

```bash
REACHY_MINI_CUSTOM_PROFILE=xiaobai_app_pack_r1
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./xiaobai_app_pack/profiles/official_app
REACHY_MINI_APP_TIMEOUT_MINUTES=0
uv run reachy-mini-conversation-app --no-camera --ui --debug
```

启动日志证明：

- 加载 external profile：`xiaobai_app_pack_r1`
- 加载 external prompt。
- realtime session 初始化为 `profile='xiaobai_app_pack_r1'`
- 启动问候 transcript：`你好呀我是小白已经准备好啦`
- UI `/status` 返回 `backend_connected=true`
- UI `/ready` 返回 `ready=true`

工具白名单：

```text
play_emotion
stop_emotion
idle_do_nothing
move_head
forget
task_status
task_cancel
```

确认未启用：

- `remember`
- `camera`

安全动作：

```text
move_head left  -> {'status': 'looking left'}
move_head front -> {'status': 'looking front'}
```

真实输出自检：

- `passed=true`
- `test_sound.status=ok`
- `uploaded_tone.played.status=ok`
- `motion.passed=true`

回滚：

不设置 external profile 环境变量时，官方 app 回到内置 `profiles/`，`custom_profile=None`，`default` profile 存在。

核心运行时：

最新提交 `9b1b5cb Add Xiaobai app pack R1` 只新增 `xiaobai_app_pack/`，没有修改 `src/`。

## 风险

1. SDK `1.8.3` 与 Reachy daemon `1.8.0` 不一致，长跑前建议升级或记录兼容风险。
2. 正式切官方 app 长跑前，应停止旧 `moss-ghost` screen，避免媒体和动作控制竞争。
