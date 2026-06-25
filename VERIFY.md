# 明天验证指南

## 一条命令启动

```bash
cd /Users/laodong/Documents/xiaobai-moss
/Users/laodong/Documents/MOSShell/.venv/bin/python run_xiaobai_real.py
```

## 你应该看到和听到的

### 启动阶段
1. 终端显示 "✓ Reachy Mini 就绪"
2. 小白做一个 greeting 动作（慢慢抬头看你，1.5秒）
3. 你听到 "你好呀，我是小白"（电脑 TTS）
4. 小白头回正

### 对话阶段
你打字输入后：
1. 终端显示安全等级 [🟢] 或 [🟡] 或 [🔴]
2. 终端显示 [动作: 手势名]
3. 小白做对应动作（1-2秒，缓慢平稳）
4. 你听到小白说话
5. 某些动作后小白会回正

### 手势对应关系（用眼睛验证）
- greeting → 头微微抬起偏一点（像看你）
- nod → 头低下再抬起（像点头）
- tilt_curious → 头歪向一边（像好奇）
- look_up_excited → 头明显抬起（像兴奋）
- sad → 头低下（像难过）
- thinking → 头歪+偏转（像在想）
- turn_look → 头转向左边（像注意到什么）

### 退出
输入 quit → 小白做 goodbye 动作（低头）→ 说"再见啦" → 断开连接

## 如果有问题

### 机器人没动
- 检查 Reachy Mini 是否开机联网
- 终端是否显示 "✓ Reachy Mini 就绪"

### 动作抽搐/不自然
- 可能是 SDK 版本不匹配（当前 SDK=1.8.3, daemon=1.8.0）
- 尝试在 Reachy 控制台里把 daemon 升级到 1.8.3

### 没有声音
- 检查 Mac 音量
- 声音目前从电脑喇叭出来（不是机器人）

### 模型没回复
- 检查网络连接（需要访问 DashScope API）

## 已修复的问题

### SDK 版本不匹配（已修复）
- 之前：SDK 1.8.3 vs daemon 1.8.0 → 动作异常/抽搐
- 现在：SDK 1.8.0 = daemon 1.8.0 → 匹配，无警告

### 预录动作库不兼容（已修复）
- 之前：用 RecordedMove 播放 HuggingFace 表情库 → 抽风
- 现在：只用 goto_target 做简单手势 → 安全可控

### 动作 API 调用方式（已修复）
- 之前：async_play_move + asyncio 冲突 → 动作不可预测
- 现在：goto_target 同步阻塞 → 动作完成才继续
