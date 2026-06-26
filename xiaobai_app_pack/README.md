# Xiaobai App Pack R1

小白 App Pack 是 Reachy Mini 的应用层包，不是新的机器人底层系统。

本包只放可迁移内容：

- profile / prompt
- app manifest
- tool / sidecar contract
- safety policy
- learning / memory / desktop control 数据约定

底层实时语音、摄像头、动作播放、机器人 runtime 由 host runtime 负责。R1 默认 host 是 Pollen 官方 `reachy_mini_conversation_app`。

## R1 内容

| App | 当前形态 | 是否改底层 |
| --- | --- | --- |
| Personality App | 官方 external profile | 否 |
| Learning App | 应用 contract + 课程流程 | 否 |
| Memory Candidate App | sidecar contract + 审核规则 | 否 |
| Desktop Control App | sidecar contract + 权限规则 | 否 |

## 官方 App 接入方式

将 `profiles/official_app/xiaobai_app_pack_r1` 作为 external profile root 的一个 profile 使用。

注意：官方 app 当前源码里已经有内置 `xiaobai` profile，所以外部包使用 `xiaobai_app_pack_r1`，避免和内置 profile 同名冲突。

```bash
REACHY_MINI_CUSTOM_PROFILE=xiaobai_app_pack_r1
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./xiaobai_app_pack/profiles/official_app
```

R1 不需要修改官方 app 核心代码。

## Memory Candidate sidecar

启动最小记忆候选 sidecar：

```bash
python -m xiaobai_app_pack.sidecars.memory_candidate_sidecar
```

默认地址：`http://127.0.0.1:8788`

默认数据目录：`~/.local/share/xiaobai_app_pack/memory/`

外部 profile 会通过 profile-local `memory_candidate` 工具创建候选记忆，通过 profile-local `forget` 工具删除已批准记忆。`remember` 仍不在工具白名单里。

家长审核 CLI：

```bash
python -m xiaobai_app_pack.sidecars.memory_candidate_cli candidates
python -m xiaobai_app_pack.sidecars.memory_candidate_cli approve <candidate_id>
python -m xiaobai_app_pack.sidecars.memory_candidate_cli reject <candidate_id>
python -m xiaobai_app_pack.sidecars.memory_candidate_cli approved
python -m xiaobai_app_pack.sidecars.memory_candidate_cli forget <matching_phrase>
```

CLI 只访问 sidecar API，不接触摄像头、麦克风、原始音频、人脸或声纹数据；输出会对常见 API key / token 形态做脱敏。

## 一键启动官方 App

```bash
scripts/start_xiaobai_official_app.sh
```

它会停止旧 `moss-ghost` screen、启动 Memory Candidate sidecar、设置 external profile 环境、启动 Pollen 官方 Conversation App，并等待 `/ready`、`/status` 与 Reachy 输出健康检查。

只看命令、不真实启动：

```bash
scripts/start_xiaobai_official_app.sh --dry-run --skip-output-check
```

日志默认在：`~/.local/state/xiaobai_app_pack/`

## 家庭 Alpha 验收

```bash
scripts/run_xiaobai_alpha_acceptance.sh
```

查看真人话术：

```bash
uv run python -m xiaobai_app_pack.acceptance.alpha_acceptance manual-script
```

验收包会自动检查 ready/status、profile tools、Reachy 输出、Memory Candidate + CLI approve/forget，并扫描 official app 日志里的短问答和动作请求证据。

## 验收

1. 官方 app 全量测试通过。
2. Reachy 输出自检通过：声音、上传音频、安全动作。
3. 小白 profile 可被官方 app 作为 external profile 加载。
4. 关闭本包后，官方 app 仍可回到默认 profile。
5. 本包不包含 API Key、孩子资料、人脸、声纹、原始音频。
