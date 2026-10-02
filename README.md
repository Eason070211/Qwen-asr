# Qwen3-ASR 本地部署

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的本地语音识别服务器，
运行在 **CPU** 上（AMD Ryzen 7 8745HS + Radeon 780M 核显，无 CUDA）。

## 硬件要求

| 项目 | 说明 |
|------|------|
| CPU | AMD Ryzen 7 8745HS（8C/16T） |
| GPU | Radeon 780M（核显，不参与推理） |
| 内存 | 19 GB（0.6B 模型约需 3-4 GB） |
| 磁盘 | 模型权重 ~1.9 GB + Python 依赖 ~3 GB |

## 环境（已就绪）

本项目与 YOLO 共用 conda 环境 **`D:\Anaconda\envs\yolo-asr`**（Python 3.12），
依赖已全部装好（torch 2.14.0+cpu / transformers 4.57.6 / qwen-asr 0.0.6 / fastapi / uvicorn 等）。

也可以按 `setup_env.ps1` 新建独立 `.venv`（包源默认官方 PyPI，
国内网络不可达时传 `-Mirror "https://pypi.tuna.tsinghua.edu.cn/simple"`）。

常用入口（先执行一次 `cd E:\YOLO; .\env.ps1` 可设好缓存目录）：
```powershell
& D:\Anaconda\envs\yolo-asr\python.exe E:\YOLO\qwen3-asr\check_env.py
& D:\Anaconda\envs\yolo-asr\python.exe E:\YOLO\qwen3-asr\server.py
```

## 模型权重

默认使用 **原生格式** 仓库 `Qwen/Qwen3-ASR-0.6B`（与 qwen-asr 0.0.6 + transformers 4.x 匹配）。
权重已下载到 `models\Qwen3-ASR-0.6B\`。

下载/更新（Hugging Face 官方源；国内网络不可达时加 `--source modelscope`）：
```powershell
& D:\Anaconda\envs\yolo-asr\python.exe download_model.py --model qwen3-asr-0.6b
& D:\Anaconda\envs\yolo-asr\python.exe download_model.py --list   # 查看可选模型
```

> **注意**：`models\Qwen3-ASR-0.6B-hf` 目录是 transformers 5.x 格式（`Qwen/Qwen3-ASR-0.6B-hf`），
> 与本环境的 qwen-asr 0.0.6 **不兼容**，仅作备份保留，服务器不会加载它。

### 3. 检查环境

```powershell
& D:\Anaconda\envs\yolo-asr\python.exe check_env.py
& D:\Anaconda\envs\yolo-asr\python.exe check_env.py --quick-test   # 加载模型并跑一次推理（约 30-60s）
```

### 4. 启动服务器

```powershell
& D:\Anaconda\envs\yolo-asr\python.exe server.py
```

服务器启动后监听 `http://0.0.0.0:8910`，API 文档在 `http://localhost:8910/docs`。

## 语音输入（全局快捷键）

一个常驻小工具：**按住快捷键说话，松开自动转写**，结果复制到剪贴板（可自动粘贴）。

### 启动方式

| 方式 | 操作 |
|------|------|
| 桌面双击 | 双击 `C:\Users\Eason\Desktop\语音输入.bat` |
| 开机自启 | 已放入启动文件夹 `...\Start Menu\Programs\Startup\Qwen3-ASR语音输入.bat` |
| 手动 | `& D:\Anaconda\envs\yolo-asr\python.exe E:\YOLO\qwen3-asr\start_voice_input.bat` |

bat 会自动检查/启动 ASR 服务器（首次约 1-2 分钟加载模型），就绪后进入热键监听。

### 使用

1. 按住**右 Ctrl** 开始录音（有提示音）——单独一个键，不与任何系统快捷键冲突
2. 松开右 Ctrl → 自动转写 → 文本进剪贴板
3. 在目标窗口按 Ctrl+V 粘贴（或加 `--paste` 自动粘贴）

> 注：原先默认的 Ctrl+Alt+Space 与 Windows 系统快捷键 Alt+Space（窗口菜单）冲突，已改为右 Ctrl。

### 常用参数

```powershell
python voice_input.py --hotkey "right ctrl"       # 默认：右 Ctrl 单键
python voice_input.py --hotkey "ctrl+alt+f1"      # 或换成任意组合键
python voice_input.py --language Chinese          # 指定语言（默认自动识别）
python voice_input.py --paste                     # 转写后自动粘贴到当前窗口
python voice_input.py --test-file a.wav           # 离线测试（不依赖麦克风/热键）
```

### 取消开机自启

删除 `...\Start Menu\Programs\Startup\Qwen3-ASR语音输入.bat` 即可。

## API 接口

### OpenAI 兼容接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/transcriptions` | POST | 语音转文字（兼容 Whisper API） |
| `/v1/audio/translations` | POST | 语音翻译（强制英文输出） |
| `/v1/chat/completions` | POST | 多模态聊天（支持 audio_url） |
| `/v1/models` | GET | 模型列表 |

#### 调用示例（curl）

```bash
# 文件上传转写
curl -X POST http://localhost:8910/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "language=Chinese"

# 带时间戳（SRT 格式）
curl -X POST http://localhost:8910/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=srt"

# URL 转写
curl -X POST http://localhost:8910/v1/audio/transcriptions \
  -F "audio_url=https://example.com/speech.wav" \
  -F "model=qwen3-asr-0.6b"
```

#### Python 调用

```python
import requests

resp = requests.post(
    "http://localhost:8910/v1/audio/transcriptions",
    files={"file": open("speech.wav", "rb")},
    data={"model": "qwen3-asr-0.6b", "language": "Chinese"},
)
print(resp.json())
```

### 原生接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/transcribe` | POST | 原生转写（支持文件/URL，可选时间戳） |
| `/health` | GET | 健康检查 |

### WebSocket 流式接口

```
ws://localhost:8910/v1/audio/stream
```

客户端流程：
1. 发送 JSON 配置：`{"language": "Chinese", "sample_rate": 16000}`
2. 发送二进制 PCM 数据（16-bit signed LE, mono, 16kHz）
3. 发送 `{"action": "flush"}` 强制输出当前缓冲
4. 发送 `{"action": "end"}` 结束

服务端返回：
```json
{"type": "transcript", "text": "...", "language": "Chinese", "is_final": false}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_ASR_MODEL_PATH` | `./models/Qwen3-ASR-0.6B` | 模型路径 |
| `QWEN_ASR_PORT` | `8910` | 监听端口 |
| `QWEN_ASR_HOST` | `0.0.0.0` | 监听地址 |
| `QWEN_ASR_DTYPE` | `float32` | 推理精度 |
| `QWEN_ASR_MAX_NEW_TOKENS` | `512` | 最大生成 token |
| `QWEN_ASR_FORCED_ALIGNER_PATH` | (空) | 时间戳对齐器路径 |

## 文件结构

```
E:\YOLO\qwen3-asr\
├── models/                 # 模型权重
│   ├── Qwen3-ASR-0.6B/        # 原生格式，服务器默认加载
│   └── Qwen3-ASR-0.6B-hf/     # 仅备份（需 transformers 5.x，不兼容当前环境）
├── config.py               # 服务器配置
├── audio.py                # 音频解码
├── engine.py               # 推理引擎
├── server.py               # FastAPI 服务器
├── download_model.py       # 模型下载工具（HF 优先，ModelScope 备用）
├── check_env.py            # 环境检查
├── setup_env.ps1           # 环境搭建脚本（官方 PyPI）
├── requirements.txt        # 依赖清单
└── README.md               # 本文档
```

## 性能预期

| 场景 | 0.6B (CPU) | 1.7B (CPU) |
|------|------------|------------|
| 10s 音频转写 | ~3-5s | ~10-20s |
| 1分钟音频 | ~20-30s | ~60-120s |
| 并发能力 | 1-2 请求 | 1 请求 |

## 已知限制

- 无 GPU 加速，推理较慢（0.6B 可用，1.7B 很慢）
- 权重从 Hugging Face 官方源下载；国内网络不可达时用 `--source modelscope`
- `models/Qwen3-ASR-0.6B-hf` 为 transformers 5.x 格式，与 qwen-asr 0.0.6 不兼容（仅备份）
- `sox` Python 包不需要（qwen-asr 从不导入它）
- 流式模式在 CPU 上需要足够的算力，可能有延迟

## 故障排除

```powershell
# 依赖问题
& D:\Anaconda\envs\yolo-asr\python.exe -m pip check

# 重新搭建环境（会重建 .venv，需先激活）
.\setup_env.ps1 -Recreate

# 下载失败？检查网络
& D:\Anaconda\envs\yolo-asr\python.exe -c "import urllib.request; urllib.request.urlopen('https://huggingface.co/', timeout=10); print('OK')"
```
