# Xiaobai App Pack R1-C Memory Review CLI 验收

时间：2026-06-26 12:55 Asia/Shanghai

结果：通过。

交付：

- `xiaobai_app_pack/sidecars/memory_candidate_cli.py`
- `xiaobai_app_pack/sidecars/memory_candidate_client.py`
- `tests/test_xiaobai_memory_candidate_sidecar.py`

验收：

- CLI 能列出 candidates。
- CLI 能 approve / reject / delete candidate。
- CLI 能列出 approved memory。
- CLI 能 forget approved memory。
- CLI 不接触摄像头、麦克风、原始音频、人脸或声纹数据。
- CLI 输出会脱敏常见 API key / token。

测试：

```text
tests/test_xiaobai_memory_candidate_sidecar.py: 4 passed
tests/: 254 passed
```
