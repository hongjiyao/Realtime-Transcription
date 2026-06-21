import os
import sys
import json
import threading

import asyncio
import logging
import warnings

# ============================================================
# 环境初始化（必须在 import torch 前执行 os.environ 设置）
# ============================================================
# P-13: 延迟到 init_environment() 中设置，避免副作用导入
# 以下 os.environ 设置必须在 import torch 前完成
# 注意：setdefault 不会覆盖已设置的值，因此多次调用是安全的

# 已初始化标志，防止重复调用
_environment_initialized = False
_env_vars_set = False

def _set_env_vars():
    """设置进程级环境变量（必须在 import torch 前执行）"""
    global _env_vars_set
    if _env_vars_set:
        return
    os.environ.setdefault("MODELSCOPE_DISABLE_REMOTE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TRITON_DISABLE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _env_vars_set = True

# 模块加载时立即设置环境变量（torch 可能在此模块之后被其他模块导入）
_set_env_vars()

# 项目根目录和模型目录
_MODEL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_MODEL_BASE, "models")
_SPEAKER_BANK_DIR = os.path.join(_MODEL_BASE, "speaker_banks")
_MAX_FRAME_BYTES = 1 << 20   # 1MB — 音频帧大小上限（base64 解码后 / 二进制）
_TIMESTAMP_SEC_THRESHOLD = 1000  # 时间戳小于此值视为秒，否则视为毫秒

def init_environment():
    """初始化进程级运行环境。启动时自动调用，测试时可重置后再次调用。"""
    global _environment_initialized
    if _environment_initialized:
        return
    # Windows 低延迟优化：提升定时器精度从 15.6ms 到 1ms
    if sys.platform == "win32":
        try:
            import ctypes
            _winmm = ctypes.windll.winmm
            _winmm.timeBeginPeriod(1)
            print("[INFO] Windows timer resolution set to 1ms (low-latency mode)")
        except Exception as e:
            print(f"[WARN] Failed to set Windows timer resolution: {e}")
    # 统一日志配置：所有日志输出到终端
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        force=True,
    )
    # Windows 上 triton 不可用导致 "triton not found" 警告，过滤之
    if sys.platform == "win32":
        warnings.filterwarnings("ignore", message=".*triton not found.*")
    _environment_initialized = True

# 模块加载时自动调用
init_environment()

from dataclasses import dataclass
from typing import Optional, Any

try:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    rich_transcription_postprocess = None

try:
    import torch
    TORCH_AVAILABLE = True
    # GPU 推理不需要 CPU 多线程并行，inter-op 和 intra-op 均设为 1
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as e:
        print(f"[WARN] Failed to set interop threads: {e}")
    torch.set_num_threads(1)
except ImportError:
    TORCH_AVAILABLE = False

try:
    from funasr import AutoModel
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False

try:
    from server.audio_capture import WasapiAudioCapture
    WASAPI_AVAILABLE = True
except ImportError:
    try:
        from audio_capture import WasapiAudioCapture
        WASAPI_AVAILABLE = True
    except ImportError:
        WASAPI_AVAILABLE = False

# Paraformer-large (vocab8404) 实际支持的语言：
# 中文（普通话、粤语、吴语、闽南语、客家话、赣语、湘语、晋语及多种口音）+ 英文
# 如需多语言支持，需切换到 paraformer-mtl 或 fun-asr-mtl 模型
LANG_MAP = {
    "zh": "中文", "en": "English",
    "auto": None,
}

SAMPLE_RATE = 16000

# 流式识别配置: [左上下文, 当前chunk, 右上下文]（帧数，每帧=60ms）
# 默认 5 帧=300ms，平衡延迟和 GPU 利用率；可通过前端面板动态调整
DEFAULT_CHUNK_SIZE_FRAMES = 5
STREAM_ENCODER_LOOK_BACK = 2
STREAM_DECODER_LOOK_BACK = 1

# VAD 端点检测配置
VAD_CHECK_INTERVAL_SEC = 0.15   # VAD 检测间隔（秒），低延迟优化：150ms
VAD_SILENCE_GAP_SEC = 0.5      # 语音段最大静音间隔（秒），超过则触发 Pass2

# Pass2 可用离线模型列表
# latency_rank: 延迟排名（1=最快），基于 GPU 推理速度
# latency_desc: 延迟描述
# lang_options: 该模型支持的语言选项列表（value → 显示名）
# local_dir: 扁平目录名，对应 models/<local_dir>/
OFFLINE_MODELS = {
    "sensevoice": {
        "id": "iic/SenseVoiceSmall",
        "local_dir": "sensevoice",
        "name": "SenseVoice-Small",
        "desc": "中英日韩粤，情感+事件检测",
        "latency_rank": 1,
        "latency_desc": "极快 (~73ms/8s, 无标点/说话人)",
        "latency_info": {"badge": "badge-fast", "desc": "低延迟，多语言"},
        "trust_remote_code": True,
        "supports_spk": False,
        "supports_timestamps": False,
        "supports_hotwords": False,
        "lang_options": [
            ("auto", "Auto"), ("zh", "中文"), ("en", "English"),
            ("ja", "日本語"), ("ko", "한국어"), ("yue", "粤语"),
        ],
    },
    "seaco_paraformer": {
        "id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "local_dir": "seaco-paraformer",
        "name": "SeACoParaformer",
        "desc": "中英，热词+说话人+时间戳",
        "latency_rank": 2,
        "latency_desc": "很快 (~176ms/8s, 含标点+说话人)",
        "latency_info": {"badge": "badge-medium", "desc": "中等延迟，支持热词"},
        "trust_remote_code": False,
        "supports_spk": True,
        "supports_timestamps": True,
        "supports_hotwords": True,
        "lang_options": [
            ("auto", "Auto"), ("zh", "中文"), ("en", "English"),
        ],
    },
    "funasr_nano": {
        "id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "local_dir": "funasr-nano",
        "name": "Fun-ASR-Nano",
        "desc": "中英日，LLM 800M",
        "latency_rank": 4,
        "latency_desc": "中等 (~770ms/8s, LLM自回归解码)",
        "latency_info": {"badge": "badge-fast", "desc": "低延迟，轻量级"},
        "trust_remote_code": True,
        "supports_spk": True,
        "supports_timestamps": False,
        "supports_hotwords": True,
        "lang_options": [
            ("auto", "Auto"), ("zh", "中文"), ("en", "English"), ("ja", "日本語"),
        ],
    },
    "funasr_nano_mlt": {
        "id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "local_dir": "funasr-nano-mlt",
        "name": "Fun-ASR-MLT-Nano",
        "desc": "31种语言，LLM 800M",
        "latency_rank": 3,
        "latency_desc": "中等 (~763ms/8s, LLM自回归解码)",
        "latency_info": {"badge": "badge-fast", "desc": "低延迟，多语言"},
        "trust_remote_code": True,
        "supports_spk": True,
        "supports_timestamps": False,
        "supports_hotwords": True,
        "lang_options": [
            ("auto", "Auto"), ("zh", "中文"), ("en", "English"), ("yue", "粤语"),
            ("ja", "日本語"), ("ko", "한국어"), ("vi", "Tiếng Việt"),
            ("id", "Bahasa Indonesia"), ("th", "ไทย"), ("ms", "Bahasa Melayu"),
            ("fil", "Filipino"), ("ar", "العربية"), ("hi", "हिन्दी"),
            ("de", "Deutsch"), ("fr", "Français"), ("es", "Español"),
            ("pt", "Português"), ("ru", "Русский"), ("it", "Italiano"),
            ("nl", "Nederlands"), ("pl", "Polski"), ("sv", "Svenska"),
            ("cs", "Čeština"), ("el", "Ελληνικά"), ("fi", "Suomi"),
            ("bg", "Български"), ("hr", "Hrvatski"), ("da", "Dansk"),
            ("et", "Eesti"), ("hu", "Magyar"), ("ro", "Română"),
            ("sk", "Slovenčina"), ("sl", "Slovenščina"), ("lt", "Lietuvių"),
            ("lv", "Latviešu"), ("mt", "Malti"), ("ga", "Gaeilge"),
        ],
    },
}

# 模型路径常量
# 模型本地路径（扁平目录结构：models/<dir>/）
MODEL_ONLINE = os.path.join(_MODELS_DIR, "paraformer-online")
MODEL_VAD = os.path.join(_MODELS_DIR, "vad")
MODEL_PUNC = os.path.join(_MODELS_DIR, "punc")
MODEL_SPK = os.path.join(_MODELS_DIR, "campplus")

# 全局可配置参数（VAD 由模型控制，不再需要静音阈值）
_CONFIG_FILE_PATH = os.path.join(_MODEL_BASE, "config.json")

def _normalize_timestamp_ms(value) -> float:
    """将时间戳归一化为毫秒：float 且小于阈值视为秒，乘以 1000"""
    if isinstance(value, float) and value < _TIMESTAMP_SEC_THRESHOLD:
        return value * 1000
    return float(value)

@dataclass
class AppConfig:
    """应用配置单例，支持文件持久化

    P05: 线程安全 — 所有字段读写通过 _lock 保护。
    P04: 防抖写入 — update() 不直接写文件，而是设置防抖定时器（1秒），
         合并高频配置更新为单次磁盘 I/O。
    """
    max_sentence_chars: int = 100
    vad_silence_gap_sec: float = 0.5
    vad_check_interval_sec: float = 0.15
    preset_spk_num: int = 2  # 指定说话人数量（最小1）
    spk_sim_threshold: float = 0.65  # 说话人匹配余弦相似度阈值（0.01-0.99，越高越严格）
    spk_ema_alpha: float = 0.3  # 说话人中心更新系数（0.01-0.99，越大越快适应）
    chunk_size_frames: int = DEFAULT_CHUNK_SIZE_FRAMES  # Pass1 chunk 帧数（5=300ms, 10=600ms, 15=900ms）

    _CONFIG_FILE = _CONFIG_FILE_PATH
    _lock: threading.Lock = threading.Lock()
    _save_pending: bool = False
    _save_timer: object = None  # asyncio.TimerHandle

    @staticmethod
    def _safe_float(val, default):
        """安全转换 float，过滤 NaN/Inf，失败返回 default"""
        try:
            v = float(val)
            if v != v or abs(v) == float('inf'):  # NaN or Inf
                return default
            return v
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val, default):
        """安全转换 int，失败返回 default"""
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def update(self, config: dict) -> None:
        """P04+P05: 线程安全更新 + 防抖持久化"""
        with self._lock:
            if "max_sentence_chars" in config:
                self.max_sentence_chars = max(10, min(200, self._safe_int(config["max_sentence_chars"], self.max_sentence_chars)))
            if "vad_silence_gap_sec" in config:
                self.vad_silence_gap_sec = max(0.2, min(5.0, self._safe_float(config["vad_silence_gap_sec"], self.vad_silence_gap_sec)))
            if "vad_check_interval_sec" in config:
                self.vad_check_interval_sec = max(0.1, min(2.0, self._safe_float(config["vad_check_interval_sec"], self.vad_check_interval_sec)))
            if "preset_spk_num" in config:
                self.preset_spk_num = max(1, min(20, self._safe_int(config["preset_spk_num"], self.preset_spk_num)))
            if "spk_sim_threshold" in config:
                self.spk_sim_threshold = max(0.01, min(0.99, round(self._safe_float(config["spk_sim_threshold"], self.spk_sim_threshold), 2)))
            if "spk_ema_alpha" in config:
                self.spk_ema_alpha = max(0.01, min(0.99, round(self._safe_float(config["spk_ema_alpha"], self.spk_ema_alpha), 2)))
            if "chunk_size_frames" in config:
                val = self._safe_int(config["chunk_size_frames"], self.chunk_size_frames)
                # FunASR chunk_size 必须是 5 的倍数，钳制到最近的合法值
                val = max(5, min(30, val))
                val = (val // 5) * 5
                self.chunk_size_frames = val
            self._clamp_fields()
        self._schedule_save()

    def to_dict(self) -> dict:
        return {
            "max_sentence_chars": self.max_sentence_chars,
            "vad_silence_gap_sec": self.vad_silence_gap_sec,
            "vad_check_interval_sec": self.vad_check_interval_sec,
            "preset_spk_num": self.preset_spk_num,
            "spk_sim_threshold": self.spk_sim_threshold,
            "spk_ema_alpha": self.spk_ema_alpha,
            "chunk_size_frames": self.chunk_size_frames,
        }

    def save_to_file(self) -> None:
        """将配置保存到 JSON 文件（P05: 线程安全）"""
        with self._lock:
            try:
                with open(self._CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Failed to save config to {self._CONFIG_FILE}: {e}")

    def _schedule_save(self) -> None:
        """P04: 防抖写入 — 1秒内多次 update 合并为单次磁盘 I/O。

        优先尝试 asyncio 定时器（在事件循环内调用时）；若事件循环不可用
        （如测试环境），则立即同步写入。
        """
        # 取消上一次挂起的定时器
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None

        try:
            loop = asyncio.get_running_loop()
            self._save_timer = loop.call_later(1.0, self._debounced_save)
        except RuntimeError:
            # 无事件循环时直接写入
            self.save_to_file()

    def _debounced_save(self) -> None:
        """防抖定时器回调"""
        self._save_timer = None
        self.save_to_file()

    def _clamp_fields(self):
        """将所有字段钳制到合法范围（消除 update / load_from_file 中的重复钳制逻辑）"""
        self.max_sentence_chars = max(10, min(200, self.max_sentence_chars))
        self.vad_silence_gap_sec = max(0.2, min(5.0, self.vad_silence_gap_sec))
        self.vad_check_interval_sec = max(0.1, min(2.0, self.vad_check_interval_sec))
        self.preset_spk_num = max(1, min(20, self.preset_spk_num))
        self.spk_sim_threshold = max(0.01, min(0.99, self.spk_sim_threshold))
        self.spk_ema_alpha = max(0.01, min(0.99, self.spk_ema_alpha))
        val = max(5, min(30, self.chunk_size_frames))
        self.chunk_size_frames = (val // 5) * 5

    def load_from_file(self) -> None:
        """从 JSON 文件加载配置，文件不存在时静默跳过（P05: 线程安全）"""
        try:
            if os.path.exists(self._CONFIG_FILE):
                with open(self._CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                with self._lock:
                    # 类型感知转换，避免 JSON 中的错误类型绕过 dataclass 校验
                    type_map = {
                        'max_sentence_chars': self._safe_int,
                        'vad_silence_gap_sec': self._safe_float,
                        'vad_check_interval_sec': self._safe_float,
                        'preset_spk_num': self._safe_int,
                        'spk_sim_threshold': self._safe_float,
                        'spk_ema_alpha': self._safe_float,
                        'chunk_size_frames': self._safe_int,
                    }
                    for key, val in data.items():
                        if hasattr(self, key) and not key.startswith('_'):
                            converter = type_map.get(key)
                            if converter:
                                setattr(self, key, converter(val, getattr(self, key)))
                            else:
                                setattr(self, key, val)
                    self._clamp_fields()
                print(f"[INFO] Config loaded from {self._CONFIG_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to load config from {self._CONFIG_FILE}: {e}")

@dataclass
class ModelCache:
    """模型缓存单例"""
    stream_model: Any = None
    stream_model_device: Optional[str] = None
    vad_model: Any = None
    vad_model_device: Optional[str] = None
    offline_model: Any = None
    offline_model_key: Optional[str] = None
    offline_model_device: Optional[str] = None
    offline_model_spk: bool = False
    current_offline_model_key: str = "seaco_paraformer"
    current_device: str = "auto"
    current_audio_device_index: Optional[int] = None

_app_config = AppConfig()
_model_cache = ModelCache()

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
