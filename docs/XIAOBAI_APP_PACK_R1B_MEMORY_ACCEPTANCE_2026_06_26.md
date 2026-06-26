# Xiaobai App Pack R1-B Memory Candidate 验收

时间：2026-06-26 12:24-12:36 Asia/Shanghai

## 结果

通过。

## 交付

- `xiaobai_app_pack/sidecars/memory_candidate_sidecar.py`
- `xiaobai_app_pack/sidecars/memory_candidate_client.py`
- `xiaobai_app_pack/profiles/official_app/xiaobai_app_pack_r1/memory_candidate.py`
- `xiaobai_app_pack/profiles/official_app/xiaobai_app_pack_r1/forget.py`
- `tests/test_xiaobai_memory_candidate_sidecar.py`

## 关键证据

- `remember` 不在 profile tools 白名单。
- `memory_candidate` 是 profile-local tool。
- `forget` 是 profile-local tool，用于删除 approved sidecar memory。
- sidecar API 支持：
  - `GET /health`
  - `POST /memory/candidates`
  - `POST /memory/candidates/{id}/approve`
  - `POST /memory/candidates/{id}/reject`
  - `DELETE /memory/candidates/{id}`
  - `GET /memory/approved`
  - `POST /memory/forget`
  - `DELETE /memory/{id}`

官方真实进程日志确认：

```text
Loaded external profile tool: memory_candidate
Loaded external profile tool: forget
Tools to be used in conversation: ['play_emotion', 'stop_emotion', 'idle_do_nothing', 'move_head', 'memory_candidate', 'forget', 'task_status', 'task_cancel']
```

sidecar 停止时：

- `memory_candidate` 返回 `sidecar_unavailable=true`
- 官方 app `/status` 返回 `backend_connected=true`
- 官方 app `/ready=true`
- 启动问候 transcript：`你好我是小白已经准备好啦`
- `move_head` 仍可执行

测试：

```text
tests/test_xiaobai_memory_candidate_sidecar.py: 3 passed
tests/: 253 passed
```

## 现场说明

旧 `moss-ghost` screen 已停止，避免和官方 app 抢 media/action。Reachy daemon 仍是 `1.8.0`，官方 SDK 是 `1.8.3`，长跑前建议升级或记录兼容风险。
