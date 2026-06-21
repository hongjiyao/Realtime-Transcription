import asyncio
import time
import traceback
import re
import threading
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from server.config import (
    TORCH_AVAILABLE, ASR_AVAILABLE, WASAPI_AVAILABLE,
    SAMPLE_RATE, LANG_MAP, STREAM_ENCODER_LOOK_BACK, STREAM_DECODER_LOOK_BACK,
    _MAX_FRAME_BYTES, _normalize_timestamp_ms, _app_config,
    _model_cache, rich_transcription_postprocess,
)
from server.models import (
    _gpu_executor, _pass2_executor,
    _ensure_executors,
    _cuda_stream_pass1, _cuda_stream_vad, _cuda_stream_pass2,
    _gpu_event_pass1, _gpu_event_vad, _gpu_event_pass2,
    _get_model_lock, _get_config_lock,
    _resolve_device, _check_model_available,
    get_or_create_models, clear_model_cache,
)
from server import pipeline, vad
from server.speaker import SpeakerBank

try:
    from server.audio_capture import WasapiAudioCapture
except ImportError:
    try:
        from audio_capture import WasapiAudioCapture
    except ImportError:
        WasapiAudioCapture = None

_AUDIO_BUFFER_SIZE = 160000
_SPEECH_BUFFER_SIZE = 160000
_VAD_BUFFER_SIZE = 48000


class RingBuffer:
    """预分配环形缓冲区，替代 np.concatenate 实现零分配读写。"""
    def __init__(self, capacity: int = 480000, dtype: type = np.float32) -> None:
        self.buf = np.zeros(capacity, dtype=dtype)
        self.capacity = capacity
        self._write_pos: int = 0
        self._size: int = 0
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def write(self, data: np.ndarray) -> None:
        with self._lock:
            n = len(data)
            if n > self.capacity:
                data = data[-self.capacity:]
                n = self.capacity

            first = min(n, self.capacity - self._write_pos)
            self.buf[self._write_pos:self._write_pos + first] = data[:first]
            if first < n:
                self.buf[:n - first] = data[first:]

            self._write_pos = (self._write_pos + n) % self.capacity
            self._size = min(self._size + n, self.capacity)

    def _read_contiguous(self, start: int, n: int) -> np.ndarray:
        if start + n <= self.capacity:
            return self.buf[start:start + n].copy()
        else:
            first = self.capacity - start
            result = np.empty(n, dtype=self.buf.dtype)
            result[:first] = self.buf[start:]
            result[first:] = self.buf[:n - first]
            return result

    def read(self, n: int) -> np.ndarray:
        with self._lock:
            if n < 0:
                raise ValueError("read count cannot be negative")
            if n > self._size:
                n = self._size
            if n == 0:
                return np.array([], dtype=self.buf.dtype)

            start = (self._write_pos - self._size) % self.capacity
            result = self._read_contiguous(start, n)

            self._size -= n
            return result

    def peek(self, n: int) -> np.ndarray:
        with self._lock:
            if n > self._size:
                n = self._size
            if n == 0:
                return np.array([], dtype=self.buf.dtype)

            start = (self._write_pos - self._size) % self.capacity
            return self._read_contiguous(start, n)

    def peek_view(self, n: int) -> np.ndarray:
        with self._lock:
            n = min(n, self._size)
            if n <= 0:
                return np.empty(0, dtype=self.buf.dtype)
            start = (self._write_pos - self._size) % self.capacity
            if start + n <= self.capacity:
                return self.buf[start:start + n]
            else:
                return self._read_contiguous(start, n)

    def peek_from_end(self, n: int) -> np.ndarray:
        with self._lock:
            if n > self._size:
                n = self._size
            if n == 0:
                return np.array([], dtype=self.buf.dtype)
            start = (self._write_pos - n) % self.capacity
            return self._read_contiguous(start, n)

    def consume(self, n: int) -> None:
        with self._lock:
            if n < 0:
                raise ValueError("consume count cannot be negative")
            if n > self._size:
                n = self._size
            self._size -= n

    def clear(self) -> None:
        with self._lock:
            self._size = 0
            self._write_pos = 0

    def copy(self) -> np.ndarray:
        with self._lock:
            return self.peek(self._size)


# ============================================================
# P-15: 策略模式注册表 — 离线模型差异处理
# 新增模型只需调用 _register_model_strategy() 注册即可，无需修改核心逻辑
# ============================================================

_OFFLINE_MODEL_STRATEGIES: dict = {}


def _register_model_strategy(model_keys, build_kwargs=None, postprocess=None):
    """注册离线模型策略

    Args:
        model_keys: 模型 key（str 或 tuple）
        build_kwargs: 构建推理参数的函数 (session) -> dict
        postprocess: 文本后处理函数 (text) -> str，仅当无 sentence_info 时调用
    """
    if isinstance(model_keys, str):
        model_keys = (model_keys,)
    for key in model_keys:
        _OFFLINE_MODEL_STRATEGIES[key] = {
            "build_kwargs": build_kwargs,
            "postprocess": postprocess,
        }


# --- seaco_paraformer ---
def _build_kwargs_seaco(session):
    kwargs = {"batch_size_s": 300}
    kwargs["hotwords"] = session._hotword_list if session._hotword_list else None
    return kwargs

_register_model_strategy("seaco_paraformer", build_kwargs=_build_kwargs_seaco)


# --- sensevoice ---
def _build_kwargs_sensevoice(session):
    return {"batch_size_s": 300, "language": "auto", "use_itn": True}

def _postprocess_sensevoice(text):
    if rich_transcription_postprocess is not None:
        return rich_transcription_postprocess(text)
    return re.sub(r'<\|[^|]*\|>', '', text).strip()

_register_model_strategy("sensevoice",
                         build_kwargs=_build_kwargs_sensevoice,
                         postprocess=_postprocess_sensevoice)


# --- funasr_nano / funasr_nano_mlt ---
def _build_kwargs_funasr_nano(session):
    kwargs = {"batch_size_s": 300, "cache": {}, "language": "auto", "use_itn": True}
    if session.hotwords:
        hw_list = [w.strip() for w in session.hotwords.split(",") if w.strip()]
        if hw_list:
            kwargs["hotwords"] = hw_list
    return kwargs

_register_model_strategy(("funasr_nano", "funasr_nano_mlt"),
                         build_kwargs=_build_kwargs_funasr_nano)


# ============================================================
# 离线模型推理参数构建 & 结果后处理（策略模式分发）
# ============================================================

def _build_offline_kwargs(model_key: str, session) -> dict:
    """根据模型类型构建离线推理参数（策略模式）"""
    strategy = _OFFLINE_MODEL_STRATEGIES.get(model_key)
    if strategy and strategy["build_kwargs"]:
        return strategy["build_kwargs"](session)
    return {"batch_size_s": 300}


def _postprocess_offline_result(model_key: str, res, session, spk_embedding, start_ms: int, end_ms: int) -> list:
    """根据模型类型后处理离线推理结果，返回标准化句子列表（策略模式）"""
    if not res or len(res) == 0:
        return []

    result = res[0]
    text = result.get("text", "")
    sentence_info = result.get("sentence_info", [])
    spk = result.get("spk", -1)

    if spk_embedding is not None and _cuda_stream_pass2 is not None:
        _cuda_stream_pass2.synchronize()

    new_sentences = []

    if sentence_info:
        if _model_cache.offline_model_spk and spk_embedding is not None:
            old_spk_values = [s.get("spk", "N/A") for s in sentence_info if isinstance(s, dict)]
            sentence_info = session.speaker_bank.relabel_sentences(sentence_info, spk_embedding)
            new_spk_values = [s.get("spk", "N/A") for s in sentence_info if isinstance(s, dict)]
            print(f"[INFO] SpeakerBank remap: {old_spk_values} → {new_spk_values}")
        else:
            spk_values = [s.get("spk", "N/A") for s in sentence_info if isinstance(s, dict)]
            print(f"[DEBUG] Pass2 raw spk: {spk_values}")

        new_sentences = _process_sentence_info(sentence_info, spk, start_ms)
    elif text:
        clean_text = text
        strategy = _OFFLINE_MODEL_STRATEGIES.get(model_key)
        if strategy and strategy["postprocess"]:
            clean_text = strategy["postprocess"](text)

        if clean_text:
            new_sentences = [{
                "text": clean_text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "spk": -1,
            }]

    return new_sentences


def _process_sentence_info(sentence_info: list, spk: int, start_ms: int) -> list:
    """处理包含 sentence_info 的推理结果"""
    final_sentences = []
    for s in sentence_info:
        if not isinstance(s, dict):
            continue
        s_text = s.get("text", s.get("sentence", ""))
        if not s_text:
            continue
        s_start = s.get("start", s.get("start_ms", s.get("start_idx", 0)))
        s_end = s.get("end", s.get("end_ms", s.get("end_idx", 0)))
        s_start = _normalize_timestamp_ms(s_start)
        s_end = _normalize_timestamp_ms(s_end)
        final_sentences.append({
            "text": s_text,
            "start_ms": start_ms + int(s_start),
            "end_ms": start_ms + int(s_end),
            "spk": s.get("spk", spk),
        })
    return final_sentences


class AsrSession:
    _FLOAT_SCALE = np.float32(1.0 / 32768.0)
    _SPEECH_ACTIVE_THRESHOLD_MS = 300
    _MIN_AUDIO_SAMPLES = 800

    @property
    def _chunk_stride(self) -> int:
        return _app_config.chunk_size_frames * 960

    @property
    def _chunk_size(self) -> list:
        return [0, _app_config.chunk_size_frames, 5]

    @property
    def _zero_chunk(self) -> np.ndarray:
        stride = self._chunk_stride
        if not hasattr(self, '_zero_chunk_cache') or self._zero_chunk_stride != stride:
            self._zero_chunk_cache = np.zeros(stride, dtype=np.float32)
            self._zero_chunk_stride = stride
        return self._zero_chunk_cache

    def __init__(self, args=None):
        self.args = args
        self._init_session_state()
        self._init_audio_buffers()
        self._init_pass1_state()
        self._init_vad_state()
        self._init_pass2_state()
        self._init_speaker_bank()
        self._init_websocket_state()

    def _init_session_state(self):
        self.language = getattr(self.args, "language", "中文") if self.args else "中文"
        self.hotwords = ""
        self._hotword_list = []
        self.is_active = False
        self.capture = None
        self.capture_task = None
        self._capture_data_event = asyncio.Event()

    def _init_audio_buffers(self):
        self.audio_buffer = RingBuffer(capacity=_AUDIO_BUFFER_SIZE)
        self.speech_buffer = RingBuffer(capacity=_SPEECH_BUFFER_SIZE)
        self.vad_buffer = RingBuffer(capacity=_VAD_BUFFER_SIZE)
        self.speech_start_ms = 0
        self.total_duration_ms = 0
        self.sentences = []
        self._pcm16_buffer = None
        self._float32_buffer = None

    def _init_pass1_state(self):
        self.stream_model = None
        self.vad_model = None
        self.offline_model = None
        self.stream_cache = {}
        self._stream_kwargs = None
        self._current_partial_list = []
        self._current_partial_str = ""
        self._current_partial_dirty = False
        self._current_partial_len = 0
        self._PARTIAL_THROTTLE_SEC = 0.1

    def _init_vad_state(self):
        self.vad_last_check_ms = 0
        self.in_speech = False
        self._vad_empty_count = 0
        # 最后检测到语音的绝对结束时刻（ms）；用于跨切片累积静音间隔
        self._last_voiced_end_ms = 0.0

    def _init_pass2_state(self):
        self._pass2_running = False
        self._pass2_done = asyncio.Event()
        self._pass2_done.set()
        self._pass2_task = None
        self._fallback_buffer_len = 0
        self._pass2_pending = False

    def _init_speaker_bank(self):
        self.speaker_bank = SpeakerBank()
        self._last_speaker_bank_data = None

    def _init_websocket_state(self):
        self.websocket = None
        self._last_sent_partial = ""
        self._last_latency_send_time = 0
        self._last_partial_send_time = 0
        self._ws_send_failed = False

    def set_language(self, lang: str) -> None:
        mapped = LANG_MAP.get(lang, lang)
        self.language = mapped if mapped is not None else self.language

    def set_hotwords(self, hotwords: str) -> None:
        self.hotwords = hotwords
        self._hotword_list = [w.strip() for w in hotwords.split(",") if w.strip()]

    @property
    def current_partial(self) -> str:
        if self._current_partial_dirty:
            self._current_partial_str = ''.join(self._current_partial_list)
            self._current_partial_dirty = False
        return self._current_partial_str

    @current_partial.setter
    def current_partial(self, value: str) -> None:
        if value == "":
            self._current_partial_list.clear()
            self._current_partial_str = ""
            self._current_partial_dirty = False
            self._current_partial_len = 0
        else:
            self._current_partial_list = [value]
            self._current_partial_str = value
            self._current_partial_dirty = False
            self._current_partial_len = len(value)

    async def _send_result(self, data: dict) -> None:
        if self._ws_send_failed:
            return
        try:
            if self.websocket and self.is_active:
                await self.websocket.send_json(data)
        except Exception as e:
            print(f"[WARN] Failed to send result: {e}")
            self._ws_send_failed = True

    def _build_fallback_result(self, text, start_ms, end_ms):
        return {
            "type": "final",
            "text": text,
            "sentences": [{"text": text, "start_ms": start_ms, "end_ms": end_ms, "spk": -1}],
            "source": "pass1_fallback"
        }

    # ==================== 会话生命周期 ====================

    async def start(self) -> None:
        if not ASR_AVAILABLE:
            raise RuntimeError("FunASR engine not available")
        if self.is_active and self.stream_model is not None:
            return
        _ensure_executors()
        print(f"[INFO] ASR session started: active_threads={threading.active_count()}")
        self.stream_model, self.vad_model, self.offline_model = await get_or_create_models(self.args)
        self.is_active = True
        self.audio_buffer.clear()
        self.sentences = []
        self.total_duration_ms = 0
        self.stream_cache = {}
        self._stream_kwargs = {
            "cache": self.stream_cache,
            "chunk_size": self._chunk_size,
            "encoder_chunk_look_back": STREAM_ENCODER_LOOK_BACK,
            "decoder_chunk_look_back": STREAM_DECODER_LOOK_BACK,
        }
        self.current_partial = ""
        self.speech_buffer.clear()
        self.speech_start_ms = 0
        self.vad_buffer.clear()
        self.vad_last_check_ms = 0
        self.in_speech = False
        self._vad_empty_count = 0
        self._last_voiced_end_ms = 0.0
        self._pass2_running = False
        print("[INFO] ASR session started (2pass mode)")

    async def reload_models(self) -> None:
        self.stream_model, self.vad_model, self.offline_model = await get_or_create_models(self.args)
        print("[INFO] Models reloaded (session state preserved)")

    # ==================== 音频缓冲管理 ====================

    async def process_audio(self, data: bytes) -> None:
        if not self.is_active:
            return
        if len(data) > _MAX_FRAME_BYTES:
            print(f"[WARN] Binary frame too large: {len(data)} bytes, max {_MAX_FRAME_BYTES}")
            return
        # P-20: 校验数据长度为 Int16 的整数倍
        if len(data) % 2 != 0:
            print(f"[WARN] Odd-length audio frame: {len(data)} bytes, dropping")
            return
        try:
            # 零拷贝：frombuffer 创建只读视图，不分配新内存
            pcm16_view = np.frombuffer(data, dtype=np.int16)
        except (ValueError, TypeError) as e:
            print(f"[WARN] Invalid audio frame data: {e}")
            return
        n = len(pcm16_view)
        if n == 0:
            return
        # 预分配 buffer 复用，避免每次分配
        if self._float32_buffer is None or len(self._float32_buffer) < n:
            self._float32_buffer = np.empty(n, dtype=np.float32)
        audio_float = self._float32_buffer[:n]
        # 向量化转换 Int16→Float32（np.multiply out= 避免临时数组）
        np.multiply(pcm16_view, self._FLOAT_SCALE, out=audio_float)
        # RingBuffer.write 内部拷贝，无需外部 .copy()
        self.audio_buffer.write(audio_float)
        await pipeline.try_process_chunks(self)

    async def _feed_from_capture(self, audio_float: np.ndarray) -> None:
        self.audio_buffer.write(audio_float)
        await pipeline.try_process_chunks(self)

    # ==================== VAD 端点检测 ====================

    async def check_vad_endpoint(self) -> bool:
        if len(self.vad_buffer) < int(SAMPLE_RATE * 0.5):
            return False

        loop = asyncio.get_running_loop()
        vad_check_samples = min(len(self.vad_buffer), int(SAMPLE_RATE * 1.5))
        segments = await loop.run_in_executor(_gpu_executor, vad.run_vad,
                                              self.vad_model, self.vad_buffer.peek_from_end(vad_check_samples),
                                              _cuda_stream_vad)
        self.vad_last_check_ms = self.total_duration_ms
        slice_ms = int(vad_check_samples / SAMPLE_RATE * 1000)
        endpoint, new_in_speech, new_empty, new_last_voiced = vad.process_vad_result(
            segments, slice_ms, self.in_speech, self._vad_empty_count,
            _app_config.vad_silence_gap_sec, self._SPEECH_ACTIVE_THRESHOLD_MS,
            last_voiced_end_ms=self._last_voiced_end_ms,
            current_total_ms=self.total_duration_ms,
        )
        self._last_voiced_end_ms = new_last_voiced
        self.in_speech = new_in_speech
        self._vad_empty_count = new_empty
        return endpoint

    # ==================== Pass1 流式识别 ====================

    def run_streaming_chunk(self, audio_chunk: np.ndarray, is_final: bool) -> dict:
        try:
            kwargs = {**self._stream_kwargs, "input": audio_chunk, "is_final": is_final}

            t0 = time.time()
            with torch.no_grad():
                if _cuda_stream_pass1 is not None:
                    with torch.cuda.stream(_cuda_stream_pass1):
                        res = self.stream_model.generate(**kwargs)
                    _gpu_event_pass1.record()
                else:
                    res = self.stream_model.generate(**kwargs)
            t1 = time.time()
            inference_ms = (t1 - t0) * 1000

            chunk_duration_ms = int(len(audio_chunk) / SAMPLE_RATE * 1000)
            self.total_duration_ms += chunk_duration_ms

            rtf = inference_ms / chunk_duration_ms if chunk_duration_ms > 0 else 0

            if res and len(res) > 0:
                text = res[0].get("text", "")
                if text and text.strip():
                    stripped = text.strip()
                    self._current_partial_list.append(stripped)
                    self._current_partial_dirty = True
                    self._current_partial_len += len(stripped)

            result = {
                "type": "partial",
                "text": self.current_partial,
                "duration_ms": self.total_duration_ms,
                "is_final": False,
                "latency": {
                    "pass": 1,
                    "inference_ms": round(inference_ms, 1),
                    "audio_ms": chunk_duration_ms,
                    "rtf": round(rtf, 3),
                },
            }

            if is_final:
                result["is_final"] = True
                result["text"] = self.current_partial
                print(f"[INFO] Pass1 is_final: '{self.current_partial}' (latency={inference_ms:.1f}ms)")

            return result
        except Exception as e:
            print(f"[ERROR] Pass1 streaming ASR error: {e}")
            traceback.print_exc()
            return None

    # ==================== Pass2 离线重识别 ====================

    def schedule_pass2(self, pass2_buffer_len: int = 0) -> None:
        """调度一次 Pass2。集中管理 _pass2_task，避免覆盖正在运行的任务引用。

        - 若无 Pass2 在运行：启动新任务。
        - 若已有 Pass2 在运行：不创建新任务（run_pass2_async 内部会排队重跑）。
        """
        if self._pass2_running:
            # 已在运行：交由 run_pass2_async 排队，但这里仍需触发排队逻辑，
            # 故创建一个"轻量"协程——它会立即走 already-running 分支后返回。
            task = asyncio.create_task(self.run_pass2_async(pass2_buffer_len=pass2_buffer_len))
            # 不覆盖 _pass2_task：保留对真正在跑的任务的引用，供 cleanup() 取消
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        else:
            self._pass2_task = asyncio.create_task(self.run_pass2_async(pass2_buffer_len=pass2_buffer_len))

    async def run_pass2_async(self, pass2_buffer_len=0) -> None:
        """触发一次 Pass2 离线重识别。

        若已有 Pass2 在运行，则不取消它（避免丢弃正在计算的精炼结果）：
        仅发送 Pass1 fallback 让 UI 立即看到文本，并记录一次"待处理"请求，
        待当前 Pass2 完成后再处理 speech_buffer 中剩余的新数据。
        """
        if self._pass2_running:
            print("[INFO] Pass2 already running, sending fallback and queuing re-run")
            fallback_text = getattr(self, '_finalized_text', '') or self.current_partial.strip()
            fallback_start = getattr(self, '_finalized_start_ms', 0) or self.speech_start_ms
            if fallback_text:
                await self._send_result(self._build_fallback_result(fallback_text, fallback_start, self.total_duration_ms))
            # 累计待处理的数据量，并标记需要在当前 Pass2 完成后重跑
            self._fallback_buffer_len = max(self._fallback_buffer_len, pass2_buffer_len)
            self._pass2_pending = True
            return

        self._pass2_running = True
        self._pass2_done.clear()
        try:
            await self.run_pass2(pass2_buffer_len=pass2_buffer_len)
        except asyncio.CancelledError:
            print("[INFO] Pass2 cancelled")
            self._reset_utterance(consume_len=max(pass2_buffer_len, self._fallback_buffer_len))
            self._fallback_buffer_len = 0
        except Exception as e:
            print(f"[ERROR] Pass2 async error: {e}")
            traceback.print_exc()
            self._reset_utterance(consume_len=max(pass2_buffer_len, self._fallback_buffer_len))
            self._fallback_buffer_len = 0
        finally:
            self._pass2_running = False
            self._pass2_done.set()
            # 当前 Pass2 完成后，若期间又积累了足够数据，则再跑一次
            if getattr(self, '_pass2_pending', False):
                if len(self.speech_buffer) >= int(SAMPLE_RATE * 0.3):
                    self._pass2_pending = False
                    pending_len = len(self.speech_buffer)
                    self._pass2_running = True
                    self._pass2_done.clear()
                    try:
                        await self.run_pass2(pass2_buffer_len=pending_len)
                    except Exception as e:
                        print(f"[ERROR] Pass2 pending re-run error: {e}")
                        self._reset_utterance(consume_len=pending_len)
                    finally:
                        self._pass2_running = False
                        self._pass2_done.set()
                else:
                    # buffer 不足 0.3s，保留 pending 标志和残留音频，等待下一句端点触发时合并处理
                    print(f"[DEBUG] Pending re-run deferred: speech_buffer too short ({len(self.speech_buffer)} samples)")

    async def run_pass2(self, pass2_buffer_len=0) -> None:
        if self.offline_model is None:
            print("[WARN] Pass2 skipped: offline model not loaded (switching?)")
            self._reset_utterance(consume_len=pass2_buffer_len)
            self._fallback_buffer_len = 0
            return

        if len(self.speech_buffer) < int(SAMPLE_RATE * 0.3):
            print("[DEBUG] Speech buffer too short, sending Pass1 fallback")
            fallback_text = getattr(self, '_finalized_text', '') or self.current_partial.strip()
            fallback_start = getattr(self, '_finalized_start_ms', 0) or self.speech_start_ms
            if fallback_text:
                await self._send_result(self._build_fallback_result(fallback_text, fallback_start, self.total_duration_ms))
            self._reset_utterance(consume_len=pass2_buffer_len)
            self._fallback_buffer_len = 0
            return

        audio_data = self.speech_buffer.peek(pass2_buffer_len) if pass2_buffer_len > 0 else self.speech_buffer.copy()
        start_ms = getattr(self, '_finalized_start_ms', 0) or self.speech_start_ms
        end_ms = self.total_duration_ms

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_pass2_executor, self.run_offline_asr, audio_data, start_ms, end_ms)

        if result:
            await self._send_result(result)
            print(f"[INFO] Sent final result: {len(self.sentences)} sentences total")

        self._reset_utterance(consume_len=pass2_buffer_len)
        self._fallback_buffer_len = 0

    def run_offline_asr(self, audio_data: np.ndarray, start_ms: int, end_ms: int) -> dict:
        if self.offline_model is None:
            return None
        if len(audio_data) < self._MIN_AUDIO_SAMPLES:
            return None

        try:
            model_key = _model_cache.current_offline_model_key
            audio_duration_sec = len(audio_data) / SAMPLE_RATE
            audio_duration_ms = int(audio_duration_sec * 1000)

            kwargs = _build_offline_kwargs(model_key, self)

            t0 = time.time()
            with torch.no_grad():
                if _cuda_stream_pass2 is not None:
                    with torch.cuda.stream(_cuda_stream_pass2):
                        res = self.offline_model.generate(input=audio_data, **kwargs)
                    _gpu_event_pass2.record()
                else:
                    res = self.offline_model.generate(input=audio_data, **kwargs)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            print(f"[INFO] Pass2 ({model_key}): latency={latency_ms:.1f}ms, "
                  f"audio={audio_duration_ms}ms, RTF={rtf:.3f}")

            spk_embedding = None
            if res and len(res) > 0:
                spk_embedding = res[0].get("spk_embedding", None)

            new_sentences = _postprocess_offline_result(model_key, res, self, spk_embedding, start_ms, end_ms)

            self.sentences.extend(new_sentences)

            final_text = " ".join(s["text"] for s in new_sentences) if new_sentences else ""
            final_spk = new_sentences[0]["spk"] if new_sentences else -1

            return {
                "type": "final",
                "text": final_text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "spk": final_spk,
                "sentences": new_sentences,
                "duration_ms": self.total_duration_ms,
                "model": model_key,
                "latency": {
                    "pass": 2,
                    "inference_ms": round(latency_ms, 1),
                    "audio_ms": audio_duration_ms,
                    "rtf": round(rtf, 3),
                    "model": model_key,
                },
            }
        except Exception as e:
            print(f"[ERROR] Pass2 offline ASR error: {e}")
            traceback.print_exc()
            return None

    def _reset_utterance(self, consume_len=0) -> None:
        if consume_len > 0:
            actual_consume = min(consume_len, len(self.speech_buffer))
            self.speech_buffer.consume(actual_consume)
        else:
            if len(self.speech_buffer) > 0:
                print(f"[WARN] _reset_utterance called with consume_len=0 but speech_buffer has {len(self.speech_buffer)} samples — not clearing to prevent data loss")
        # 端点已触发并处理完毕，重置绝对时间轴的语音追踪状态
        self._last_voiced_end_ms = 0.0
        self._vad_empty_count = 0

    async def _await_pass2(self, label: str = "Pass2") -> None:
        if self._pass2_running:
            print(f"[INFO] Waiting for in-progress {label} to complete...")
            try:
                await asyncio.wait_for(self._pass2_done.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                print(f"[WARN] {label} did not complete in time")

    async def stop(self) -> dict:
        if not self.is_active:
            return {"event": "stopped", "sentences": list(self.sentences), "is_final": True}
        self.is_active = False

        await self.stop_capture()

        await self._await_pass2("Pass2")

        if len(self.audio_buffer) > 0 or self.stream_cache:
            remaining = self.audio_buffer.copy()
            self.audio_buffer.clear()
            stride = self._chunk_stride
            if len(remaining) > 0:
                pad_len = (stride - len(remaining) % stride) % stride
                if pad_len > 0:
                    remaining = np.concatenate([remaining, np.zeros(pad_len, dtype=np.float32)])
                self.speech_buffer.write(remaining)
            loop = asyncio.get_running_loop()
            dummy = np.zeros(stride, dtype=np.float32)
            result = await loop.run_in_executor(_gpu_executor, self.run_streaming_chunk, dummy, True)
            if result:
                await self._send_result(result)

        if len(self.speech_buffer) >= int(SAMPLE_RATE * 0.3):
            await self.run_pass2(pass2_buffer_len=len(self.speech_buffer))

        if self.speaker_bank and len(self.speaker_bank.centers) > 0:
            try:
                self._last_speaker_bank_data = self.speaker_bank.save_to_dict()
                print(f"[INFO] Speaker bank data cached ({len(self.speaker_bank.centers)} speakers)")
            except Exception as e:
                print(f"[WARN] Failed to cache speaker bank data: {e}")
                self._last_speaker_bank_data = None
        else:
            self._last_speaker_bank_data = None

        print("[INFO] ASR session stopped")
        # P16: 线程池生命周期由 main.py lifespan 管理，session.stop() 不再销毁
        return {
            "event": "stopped",
            "sentences": list(self.sentences),  # 返回拷贝，防止外部修改内部状态
            "partial": "",
            "duration_ms": self.total_duration_ms,
            "is_final": True,
        }

    # ==================== 音频采集管理 ====================

    async def start_capture(self, device_index: Optional[int] = None) -> None:
        if not WASAPI_AVAILABLE:
            raise RuntimeError("WASAPI capture not available")
        if self.capture is not None:
            await self.stop_capture()
        self.capture = WasapiAudioCapture(device_index=device_index)
        self._capture_data_event = asyncio.Event()
        self.capture._data_ready_event = self._capture_data_event
        self.capture.start()
        self.capture_task = asyncio.create_task(self._capture_loop())
        print(f"[INFO] WASAPI capture started (device_index={device_index})")

    async def _capture_loop(self) -> None:
        capture_event = self._capture_data_event
        try:
            while self.capture is not None and self.capture.is_capturing():
                capture_event.clear()
                chunks = []
                while True:
                    audio_chunk = self.capture.read()
                    if audio_chunk is None or len(audio_chunk) == 0:
                        break
                    chunks.append(audio_chunk)

                if chunks:
                    if len(chunks) == 1:
                        await self._feed_from_capture(chunks[0])
                    else:
                        combined = np.concatenate(chunks)
                        await self._feed_from_capture(combined)
                else:
                    try:
                        await asyncio.wait_for(capture_event.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ERROR] Capture loop error: {e}")

    async def stop_capture(self) -> None:
        if self.capture is not None:
            # 在调用 capture.stop()（PortAudio native stream.close）之前同步 CUDA，
            # 避免关闭流时与 pending CUDA kernel 并发触发 0xC0000005
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            self.capture.stop()
            self.capture = None
        if self.capture_task is not None:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
            self.capture_task = None
        print("[INFO] WASAPI capture stopped")

    async def cleanup(self) -> None:
        await self._await_pass2("Pass2")
        if self._pass2_task is not None and not self._pass2_task.done():
            self._pass2_task.cancel()
            try:
                await self._pass2_task
            except asyncio.CancelledError:
                pass
            self._pass2_task = None
        await self.stop_capture()
        self.is_active = False
        self.stream_model = None
        self.vad_model = None
        self.offline_model = None
        self.audio_buffer.clear()
        self.stream_cache = {}
        self.speech_buffer.clear()
        self.vad_buffer.clear()
        self.websocket = None
