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

常用入口（先执行一次 `cd E:\Projects\YOLO; .\env.ps1` 可设好缓存目录）：
```powershell
& D:\Anaconda\envs\yolo-asr\python.exe E:\Projects\qwen3-asr\check_env.py
& D:\Anaconda\envs\yolo-asr\python.exe E:\Projects\qwen3-asr\server.py
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

一个常驻小工具：**按一下开始说话，再按一下结束**。说话过程中的停顿会被自动识别并切段，
切出来的每一段立刻进入后台队列转写，所以**转写不会阻塞录音** —— 你可以连续说下去，
文字会一段一段依次粘贴到光标处。

### 核心机制

| 机制 | 说明 |
|------|------|
| **后台队列** | 录音线程只负责切段入队，转写在别的线程做。按热键只关闭当前这一段，不会等推理。 |
| **静音切段 (VAD)** | 说话时持续监测音量，安静超过 `--silence-hold`（默认 0.65s）就把这段语音封口入队。 |
| **自动噪声校准** | 启动后先测环境噪声底（取最近 10s 的 10% 分位数，抗偶发爆音/掉采样），静音门限 = 噪声底+10dB，说话门限 = 噪声底+18dB。校准完成前用固定门限兜底。 |
| **预卷 (pre-roll)** | 每段会多带 `--preroll`（默认 0.35s）语音开始前的音频，避免第一个字被切掉。 |
| **毛刺过滤** | 短于 `--min-speech`（默认 0.30s）的响声（敲键盘、咳嗽、关门）不会被当成一句话。 |
| **音量包络** | 判定静音用"快起慢落"的包络跟随，而不是单块 RMS —— 否则正常说话的振幅每秒数次过零，会被误判成停顿、把词切成两半。**是否算"在说话"则用原始块电平判断**，因为包络释放会在正常说话中跌破门限，用它统计说话时长会永远凑不够最短语音，最后整段被丢弃。 |
| **剪贴板保护** | 粘贴前记住你原来的剪贴板内容，粘贴完成后恢复（`--no-restore-clipboard` 可关）。 |
| **结果落盘** | 每段文字追加到 `transcripts\transcripts-YYYY-MM-DD.txt`（按大小自动轮转），粘贴没打上、窗口被关、程序崩了都不丢字。 |
| **电平归一化** | 送模型前把每段拉到 `--gain-target-db`（默认 -22 dBFS），并保证不被削顶。带自动增益的麦克风（本机 AMD 阵列电平漂移达 33 dB）必须靠它才能稳定出字。 |
| **语言回退** | 自动识别返回空时，用 `--fallback-language`（默认 Chinese）重试一次。本模型对中文在自动识别下经常直接返回空字符串。 |
| **失败保音频** | 服务器挂了会重试（默认 2 次）；仍失败就把该段音频存成 `failed_audio\failed-*.wav` 并附 `.txt` 说明，之后可手动补转。 |
| **标题栏状态** | 窗口标题实时显示 `● 录音中 / … 转写中 / ○ 待机`、待处理段数、完成/失败/取消计数，不用回头看日志。 |
| **丢弃当前段** | 说错了按 `--cancel-hotkey`（如 `right alt`）立刻作废**正在录**的这段；已排队的转写不受影响。 |

> 队列语义：一个任务算"处理完"，是在**文字落盘、粘贴、失败保音频都做完之后**。
> 所以退出时打印的"队列已清空"是真的清空，不会出现最后一段文字没写进去的情况。

### 启动方式

| 方式 | 操作 |
|------|------|
| 桌面双击 | 双击 `C:\Users\Eason\Desktop\语音输入.bat` |
| 开机自启 | `...\Start Menu\Programs\Startup\Qwen3-ASR语音输入.bat`（已指向本目录的启动器） |
| 手动 | `& D:\Anaconda\envs\yolo-asr\python.exe E:\Projects\qwen3-asr\start_voice_input.bat` |

bat 会自动检查/启动 ASR 服务器（首次约 1-2 分钟加载模型），就绪后进入热键监听。

### 使用

1. **按一下右 Ctrl** 开始录音（提示音），然后正常说话
2. 说话中间**自然停顿**即可 —— 每停够 0.65s 就切出一段，提示音响一下，后台开始转写
3. 每段转写完自动**粘贴到当前光标位置**（默认开启）
4. **再按一下右 Ctrl** 结束：把最后没说完的一段也提交，并等队列转写完

> 转写进行中也可以直接再按热键开始下一轮录音，排队即可，不会互相阻塞。
> 若积压超过 5 段会在控制台提醒（文本会随后依次输出，不会丢）。
> 说错了按 `--cancel-hotkey`（默认关闭，例如设成 `right alt`）可丢弃正在录的那段。
> 切换在**松开按键**时触发，长按不会误触发、不会卡住；单次录音超过 300s 自动停止
> （`--max-seconds` 可调）。

### 结果去哪了 / 出问题怎么办

| 情况 | 你会看到 | 数据在哪 |
|------|----------|----------|
| 正常 | `[TEXT] #3 …` + `[PASTE] #3 已粘贴到光标处` | 同时追加到 `transcripts\transcripts-<日期>.txt` |
| 这段没人说话 | `[SKIP] #3 无有效语音` | 不落盘（本就是静音/噪声） |
| 声音偏轻被补了电平 | `[GAIN] 这段偏轻，已提升 N dB 再送模型` | 正常输出文字 |
| 自动识别没出字 | `[LANG] 自动识别没出文字，改用 Chinese 重试一次` | 正常输出文字 |
| 服务器没开/中途挂了 | `[RETRY]` 重试，仍失败 `[FAIL]` | 音频存到 `failed_audio\failed-<时间>-<序号>.wav`（含 `.txt` 说明），可事后补转 |
| 粘贴打到了别的窗口 | 控制台仍有 `[TEXT]` | 从 transcripts 日志里找回 |
| 一直 `[SKIP]`、识别总是"嗯。" | — | **先跑 `check_mic_levels.py`**：多半是设备选错或电平太低（见下） |
| 没听到提示音、热键没反应 | — | 看是否 `[server] 等待超时`，或麦克风被别的程序独占 |

补转已保存的失败音频：

```powershell
& D:\Anaconda\envs\yolo-asr\python.exe voice_input.py --test-file failed_audio\failed-20261014-201013-007.wav
```

### 麦克风设备与电平（重要）

这台机器有 57 个输入设备（含 Voicemeeter 虚拟通道），**不同设备的拾音电平能差 20 dB 以上**，
而 AMD 麦克风阵列还带自动增益控制，同一设备在 20 秒内电平会在 -48 ~ -15 dBFS 之间漂移。
电平越低，模型输出越敷衍（-48 dBFS 时只会回"嗯。"甚至空字符串）。

`config.py` 里已把 `voice_device` 设为 **"16"**（`Microphone Array (AMD Audio Device)`），
这是实测拾音正常的那一路。换设备/插拔硬件后设备号可能变，用这个工具重新确认：

```powershell
# 逐个设备试录，打印电平 + 服务器识别结果，并给出建议
& D:\Anaconda\envs\yolo-asr\python.exe check_mic_levels.py

# 只看电平不问服务器；或只测指定设备
... check_mic_levels.py --skip-server
... check_mic_levels.py --devices 16 1
```

程序内部的 **电平归一化** 会兜住剩下的波动（实测给轻的那段补过 +9.8 dB 才有输出），
所以正常情况下你不需要手动调增益；如果经常看到 `[GAIN]` 补超过 20 dB，说明设备本身太弱。

### 停顿 / 灵敏度调整

默认值已按"说话停顿稍长"的人调过（0.65s 停顿、0.30s 最短语音）。觉得不合适时：

```powershell
# 我停顿更长，不想被切碎 -> 加大停顿门槛
python voice_input.py --silence-hold 0.9

# 被环境噪声误触发（没人说话也切段）-> 关掉自适应，手动抬高门限
python voice_input.py --no-adaptive-noise --silence-db -34 --speech-db -28

# 说话轻 / 麦克风增益低，识别不到 -> 关掉自适应，压低门限
python voice_input.py --no-adaptive-noise --silence-db -52 --speech-db -44

# 看看当前生效的全部参数（命令行 > 环境变量 > config.py）
python voice_input.py --dump-config

# 设备多的时候选麦克风（比如选耳麦而不是机内阵列）
python voice_input.py --list-devices
python voice_input.py --device 1
python voice_input.py --device bose

# 只想要"整段录音一次转写"的老行为
python voice_input.py --no-vad
```

### 常用参数

```powershell
python voice_input.py --hotkey "right ctrl"    # 默认：右 Ctrl 单键（切换式）
python voice_input.py --hotkey "ctrl+alt+f1"   # 或换成任意组合键
python voice_input.py --language Chinese       # 指定语言（默认自动识别）
python voice_input.py --no-paste               # 关闭自动粘贴（只复制到剪贴板）
python voice_input.py --max-seconds 300        # 录音超时自动停止（秒）
python voice_input.py --test-file a.wav        # 离线测试（不依赖麦克风/热键）
```

切段 / 队列相关的完整参数：

| 参数 | 默认 | 作用 |
|------|------|------|
| `--silence-hold` | `0.65` | 安静多久算一段结束（越大越不容易被切碎） |
| `--min-speech` | `0.30` | 短于这个时长的响声忽略（敲键盘、咳嗽） |
| `--preroll` | `0.35` | 每段前面多带的音频，避免吞字 |
| `--max-segment` | `30` | 一直不停顿时的强制切分长度 |
| `--min-segment` | `0.35` | 短于这个时长的段直接丢弃 |
| `--silence-db` / `--speech-db` | `-45` / `-38` | 固定门限（自适应校准完成前也用它兜底） |
| `--adaptive-noise` / `--no-adaptive-noise` | 开 | 是否自动测量环境噪声来定门限 |
| `--vad` / `--no-vad` | 开 | 是否停顿自动切段 |
| `--queue-workers` | `1` | 后台转写线程数（服务器本身串行推理，加多一般无益） |
| `--queue-max` | `64` | 队列上限，超过则丢最旧的段 |
| `--backlog-warn` | `5` | 积压多少段时提醒 |
| `--retry` | `2` | 单段转写失败后的重试次数 |
| `--device` | 系统默认 | 输入设备编号或名称片段 |
| `--cancel-hotkey` | 空（关闭） | 丢弃正在录的那段，如 `right alt` |
| `--restore-clipboard` / `--no-restore-clipboard` | 开 | 粘贴后是否还原原剪贴板 |
| `--log-transcripts` / `--no-log-transcripts` | 开 | 是否把每段文字追加到日志 |
| `--transcripts-dir` | `.\transcripts` | 转写文字日志目录 |
| `--failed-audio-dir` | `.\failed_audio` | 失败段的音频保存目录 |
| `--status-title` / `--no-status-title` | 开 | 标题栏是否显示状态 |

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

语音输入工具的设置（均可用同名命名的 `QWEN_ASR_VOICE_*` 环境变量长期覆盖，
或按上表用命令行参数临时覆盖；默认值写在 `config.py`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_ASR_VOICE_HOTKEY` | `right ctrl` | 切换录音的热键 |
| `QWEN_ASR_VOICE_LANGUAGE` | (空) | 强制语言，空=自动识别 |
| `QWEN_ASR_VOICE_PASTE` | `true` | 转写完自动 Ctrl+V |
| `QWEN_ASR_VOICE_RESTORE_CLIPBOARD` | `true` | 粘贴后还原原剪贴板 |
| `QWEN_ASR_VOICE_MAX_SECONDS` | `300` | 单次录音超时 |
| `QWEN_ASR_VOICE_DEVICE` | (空) | 输入设备编号或名称片段 |
| `QWEN_ASR_VOICE_VAD` | `true` | 停顿自动切段 |
| `QWEN_ASR_VOICE_ADAPTIVE_NOISE` | `true` | 自动测量环境噪声定门限 |
| `QWEN_ASR_VOICE_SILENCE_DB` | `-45` | 固定模式静音门限 (dBFS) |
| `QWEN_ASR_VOICE_SPEECH_DB` | `-38` | 固定模式说话门限 (dBFS) |
| `QWEN_ASR_VOICE_SILENCE_HOLD` | `0.65` | 安静多久算一段结束（秒） |
| `QWEN_ASR_VOICE_MIN_SPEECH` | `0.30` | 短于此长度的响声忽略（秒） |
| `QWEN_ASR_VOICE_PREROLL` | `0.35` | 每段前多带的音频（秒） |
| `QWEN_ASR_VOICE_MAX_SEGMENT` | `30` | 强制切分长度（秒） |
| `QWEN_ASR_VOICE_MIN_SEGMENT` | `0.35` | 短于此长度的段丢弃（秒） |
| `QWEN_ASR_VOICE_QUEUE_WORKERS` | `1` | 后台转写线程数 |
| `QWEN_ASR_VOICE_QUEUE_MAX` | `64` | 队列上限 |
| `QWEN_ASR_VOICE_BACKLOG_WARN` | `5` | 积压提醒阈值 |
| `QWEN_ASR_VOICE_RETRY` | `2` | 单段失败重试次数 |
| `QWEN_ASR_VOICE_LOG_TRANSCRIPTS` | `true` | 是否写转写文字日志 |
| `QWEN_ASR_VOICE_TRANSCRIPTS_DIR` | `./transcripts` | 文字日志目录 |
| `QWEN_ASR_VOICE_FAILED_AUDIO_DIR` | `./failed_audio` | 失败段音频目录 |
| `QWEN_ASR_VOICE_CANCEL_HOTKEY` | (空) | 丢弃当前段的热键 |

例：永久把停顿门槛改成 0.9s（不改代码）：

```powershell
setx QWEN_ASR_VOICE_SILENCE_HOLD 0.9
```

> `QWEN_ASR_MAX_CONCURRENT` 不会带来并行转写：推理是**严格串行**的
> （单模型实例 + 单推理线程 + 内部锁），实测 6 个并发请求是依次完成的。
> 调大它只会掩盖真实上限，`/v1/models` 里的 `inference_slots` 才是实情。

## 文件结构

```
E:\Projects\qwen3-asr\
├── models/                 # 模型权重
│   ├── Qwen3-ASR-0.6B/        # 原生格式，服务器默认加载
│   └── Qwen3-ASR-0.6B-hf/     # 仅备份（需 transformers 5.x，不兼容当前环境）
├── config.py               # 服务器 + 语音输入工具配置
├── audio.py                # 音频解码
├── engine.py               # 推理引擎（串行推理 + 内部锁）
├── server.py               # FastAPI 服务器
├── voice_input.py          # 全局快捷键语音输入（队列 + 静音切段 + 结果落盘）
├── test_voice_pipeline.py  # 切段/队列/噪声自适应自测（可离线跑，含真实服务器端到端）
├── test_voice_safety.py    # 重试/落盘/失败保音频/取消 的自测
├── test_real_speech.py     # 真实人声全链路自测（SAPI 合成中文语音 → VAD 切段 → 服务器转写比对）
├── chain_diag.py           # 一键全链路诊断：麦克风 → 电平/频谱 → 切段 → 归一化 → 服务器识别
├── stress_concurrency.py   # 并发压测（验证串行语义与稳定性）
├── transcripts/            # 转写文字日志（运行时生成）
├── failed_audio/           # 转写失败保留的音频（运行时生成）
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
