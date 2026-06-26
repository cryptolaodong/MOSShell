# Xiaobai App Pack R1-F 动作请求验收收口

时间：2026-06-26 Asia/Shanghai

## 目标

只在小白应用层补强动作请求验收，不修改 Pollen 官方 Conversation App 核心 runtime。

现场动作请求：

```text
小白，请点一下头再说收到。
```

通过标准：

- official app 日志能看到用户动作请求。
- official app 日志能看到 `move_head` tool call。
- 现场能看到头部动作，且动作不明显早于语音太久。
- 验收报告里的 `interaction_log_evidence.evidence.manual_pending` 不再包含 `action_request`。

## 本轮改动

R1-E 只有 `manual_pending(action_request)`，看不出卡点。R1-F 把日志扫描拆成明确归因：

- `no_user_voice_seen`：日志没有任何用户语音。
- `microphone_did_not_capture_action_request`：有用户语音，但没有捕获到“点头/动脑袋/转头”等动作请求。
- `llm_did_not_choose_move_head`：动作请求进入日志，但模型没有调用 `move_head`。
- `tool_called_robot_motion_failed`：`move_head` 被调用，但工具或机器人运动层报错。
- `passed_move_head_tool_seen`：日志看到 `move_head` 调用。

## 当前探针结果

报告文件：

```text
/Users/laodong/.local/state/xiaobai_app_pack/r1f_action_acceptance_probe_after_diagnosis.json
```

当前自动项：

- `official_app_ready`: ok
- `profile_tools`: ok
- `reachy_output`: ok
- `memory_candidate_cli_flow`: ok

当前动作归因：

```text
action_request_diagnosis = microphone_did_not_capture_action_request
action_request_status = manual_pending
```

解释：官方 App 在线，Reachy 输出健康，但日志最近只捕获到英文环境碎片，没有捕获到中文“点头/动脑袋/转头”动作请求。因此现在不能判定为 LLM 没调工具，也不能判定为机器人动作失败。

## 下一步

站在小白旁边说：

```text
小白，请点一下头再说收到。
```

然后运行：

```bash
/Users/laodong/Documents/Reachy/scripts/run_xiaobai_alpha_acceptance.sh
```

若仍失败，看 `action_request_diagnosis`：

- `microphone_did_not_capture_action_request`：先处理官方 App 麦克风输入。
- `llm_did_not_choose_move_head`：只改小白 profile 指令，不改底层 runtime。
- `tool_called_robot_motion_failed`：先跑 R1-D 输出健康检查，再看 daemon/motor 连接。
