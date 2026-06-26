# Xiaobai App Pack R1-E 家庭 Alpha 验收包

时间：2026-06-26 Asia/Shanghai

结果：应用层验收包已落地，不修改官方 Conversation App 核心 runtime。

命令：

```bash
scripts/run_xiaobai_alpha_acceptance.sh
uv run python -m xiaobai_app_pack.acceptance.alpha_acceptance manual-script
```

覆盖：

- `/ready` 与 `/status.backend_connected`
- `xiaobai_app_pack_r1` tools 白名单
- Reachy 输出健康
- Memory Candidate 创建
- CLI `candidates` / `approve` / `forget`
- official app 日志证据
- 真人短问答、动作请求和记忆候选话术

说明：官方 app 没有稳定公开的文本注入对话接口；短问答和动作请求采用真人现场话术 + 日志扫描取证，记忆候选和家长审核 CLI 走自动验收。

本轮验证：

```text
targeted tests: 17 passed
full tests: 267 passed
report: /Users/laodong/.local/state/xiaobai_app_pack/r1e_alpha_acceptance_20260626.json
```

当前报告状态：

- official_app_ready: ok
- profile_tools: ok
- reachy_output: ok
- memory_candidate_cli_flow: ok
- interaction_log_evidence: manual_pending(action_request)
