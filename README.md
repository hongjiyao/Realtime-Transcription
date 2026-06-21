# Realtime-Transcription

基于 FunASR 的实时语音转录系统，采用 **2pass 双通道架构**，专为 Windows + GPU 环境优化。Pass1 流式低延迟识别 + Pass2 离线精炼重识别，支持说话人分离、VAD 端点检测、WASAPI 系统音频捕获。

---

## 核心特性

- **2pass 架构**：Pass1（Paraformer-online 流式）实时输出 partial 文本，Pass2（离线模型）在 VAD 端点后对完整语音段精炼重识别，输出带标点、说话人标签、时间戳的最终结果
- **三流 CUDA 并行**：Pass1 / VAD / Pass2 在独立 CUDA Stream 上真正并行执行，互不阻塞
- **WASAPI 系统音频捕获**：Windows 原生环回捕获，转录系统音频输出（无需麦克风）
- **跨语句说话人一致性**：基于 CAMP++ 192 维 embedding + 余弦相似度匹配 + EMA 中心更新
- **多 Pass2 模型动态切换**：运行时切换 4 种离线模型，自动重载
- **零分配音频管道**：RingBuffer 预分配环形缓冲区，消除热路径内存分配
- **热路径优化**：`empty_cache` 抑制、GPU P1 状态激活、Windows 1ms 定时器精度
- **后台模型预热**：服务器秒级启动，模型后台异步加载 + CUDA 预热

---

## 技术栈

| 组件 | 技术 |
|------|------|
| ASR 引擎 | FunASR 1.3.9（Paraformer / SenseVoice / SeACoParaformer / Fun-ASR-Nano） |
| Web 框架 | FastAPI 0.136.3 + Uvicorn 0.48.0 |
| 实时通信 | WebSocket（双向命令 + 音频流） |
| GPU 加速 | PyTorch（CUDA 12.4，CUDA Streams + Events） |
| 音频捕获 | PyAudioWPatch（WASAPI Loopback） |
| 前端 | 原生 HTML + CSS + JavaScript（无框架，单页应用） |
| Python | 3.11+ |

---


### 目录结构

```
Realtime-Transcription/
├── server/                          # 后端
│   ├── main.py                      # FastAPI 入口 + WebSocket 处理
│   ├── config.py                    # 配置管理 + 环境初始化
│   ├── models.py                    # 模型加载 + CUDA Streams/Events
│   ├── pipeline.py                  # 并行调度核心
│   ├── session.py                   # AsrSession + RingBuffer
│   ├── vad.py                       # VAD 端点检测
│   ├── speaker.py                   # SpeakerBank 说话人库
│   ├── audio_capture.py             # WASAPI 环回捕获
│   └── asr.py                       # 兼容性聚合模块
├── web/                             # 前端单页应用
│   ├── index.html
│   ├── app.js
│   └── style.css
├── models/                          # FunASR 模型（gitignore）
├── config.json                      # 运行时配置（gitignore）
├── speaker_banks/                   # 说话人库（gitignore）
├── requirements.txt                 # 项目依赖列表
```

---

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+
- NVIDIA GPU + CUDA 12.4（CPU 模式可选）
- 8GB+ 显存（推荐）/ 16GB+ 内存（CPU 模式）

### 安装

1. **克隆仓库**

```bash
git clone https://github.com/your-username/Realtime-Transcription.git
cd Realtime-Transcription
```

2. **创建虚拟环境**

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **安装 PyTorch（按 CUDA 版本）**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

4. **安装依赖**

```bash
pip install -r requirements.txt
```

5. **下载模型**

从 [夸克网盘](https://pan.quark.cn/s/93fb831191ef) 下载以下模型，放置到 `models/` 目录：

```
models/
├── paraformer-online/     # Pass1 流式模型
├── vad/                   # VAD 模型
├── punc/                  # 标点模型
├── campplus/              # 说话人识别模型
├── sensevoice/            # Pass2 离线模型（任选 1+）
├── seaco-paraformer/      # Pass2 离线模型（任选 1+）
├── funasr-nano/           # Pass2 离线模型（任选 1+）
└── funasr-nano-mlt/       # Pass2 离线模型（任选 1+）
```

### 启动

```bash
# 方式 1：使用启动脚本
start.bat

# 方式 2：直接运行
python -m server.main

# 方式 3：带参数启动
python -m server.main --host 0.0.0.0 --port 8000 --spk --debug
```

启动后访问 http://localhost:8000 即可使用 Web UI。

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--host` | str | `127.0.0.1` | 监听地址 |
| `--port` | int | `8000` | 监听端口 |
| `--language` | str | `中文` | 默认识别语言 |
| `--spk` | flag | `True` | 启用说话人分离（默认开启） |
| `--no-spk` | flag | - | 关闭说话人分离 |
| `--debug` | flag | `False` | 启用 `/debug/status` 调试端点 |

---

---

## 许可证

本项目仅供学习和研究使用。所使用的 FunASR 模型遵循其各自的许可证。

---

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) - 阿里达摩院语音识别框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) - WASAPI 环回音频捕获
