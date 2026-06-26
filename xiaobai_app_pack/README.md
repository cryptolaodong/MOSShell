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

## 验收

1. 官方 app 全量测试通过。
2. Reachy 输出自检通过：声音、上传音频、安全动作。
3. 小白 profile 可被官方 app 作为 external profile 加载。
4. 关闭本包后，官方 app 仍可回到默认 profile。
5. 本包不包含 API Key、孩子资料、人脸、声纹、原始音频。
