import os
import json
import argparse
import asyncio
import time
import traceback
import base64

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# 全局调试模式标志，由 --debug CLI 参数控制
_DEBUG_MODE = False

# 内部错误通用消息（不泄露异常详情/路径给客户端）
_INTERNAL_ERROR_MSG = "服务器内部错误，请查看服务器日志"

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from server.audio_capture import WasapiAudioCapture
except ImportError:
    try:
        from audio_capture import WasapiAudioCapture
    except ImportError:
        WasapiAudioCapture = None

from server.config import (
    TORCH_AVAILABLE, ASR_AVAILABLE, WASAPI_AVAILABLE,
    _MODEL_BASE, _MODELS_DIR, _SPEAKER_BANK_DIR, _MAX_FRAME_BYTES,
    OFFLINE_MODELS, WEB_DIR,
    _app_config, _model_cache,
)
from server.models import (
    _gpu_executor, _pass2_executor,
    _ensure_executors, _shutdown_executors,
    _init_cuda_streams,
    _disable_empty_cache, _enable_empty_cache,
    _model_lock, _config_lock,
    _get_model_lock, _get_config_lock,
    _resolve_device, _check_model_available,
    get_or_create_models, clear_model_cache,
    _global_args, _global_last_speaker_bank_data, _preloaded_speaker_bank,
    _speaker_bank_global_lock,
)
from server.asr import (
    RingBuffer, SpeakerBank, AsrSession,
    _build_offline_kwargs, _postprocess_offline_result, _process_sentence_info,
)

_cached_index_html = None


# P01: 模型预热状态标志（后台任务完成前为 True）
_warming_up = False


async def _background_model_warmup():
    """P01: 后台模型预加载 + CUDA 预热

    在 lifespan yield 之后的后台任务中运行，允许服务器立即接受连接，
    同时模型加载和预热在后台进行。首次 WebSocket 请求会等待预热完成。
    """
    global _warming_up
    try:
        print("[INFO] Background warmup started (server is accepting connections)")

        await get_or_create_models(_global_args)
        print("[INFO] All models preloaded successfully")

        # CUDA 预热：消除首次请求的冷启动延迟
        if _model_cache.stream_model is not None and TORCH_AVAILABLE and torch.cuda.is_available():
            print("[INFO] CUDA warmup: running dummy inference...")
            dummy_audio = np.zeros(_app_config.chunk_size_frames * 960, dtype=np.float32)
            try:
                # 并行预热三模型
                warmup_tasks = []
                warmup_tasks.append(asyncio.get_running_loop().run_in_executor(
                    _gpu_executor,
                    lambda: [_model_cache.stream_model.generate(input=dummy_audio, cache={}, is_final=True, chunk_size=[0, _app_config.chunk_size_frames, 5]) for _ in range(3)]
                ))
                if _model_cache.vad_model is not None:
                    warmup_tasks.append(asyncio.get_running_loop().run_in_executor(
                        _gpu_executor,
                        lambda: _model_cache.vad_model.generate(input=dummy_audio, batch_size=1)
                    ))
                if _model_cache.offline_model is not None:
                    warmup_tasks.append(asyncio.get_running_loop().run_in_executor(
                        _gpu_executor,
                        lambda: _model_cache.offline_model.generate(input=dummy_audio, batch_size=1)
                    ))
                await asyncio.gather(*warmup_tasks)
                torch.cuda.synchronize()
                print("[INFO] CUDA warmup completed")

                # 禁用 empty_cache 以减少热路径延迟
                _disable_empty_cache()
                print("[INFO] Disabled torch.cuda.empty_cache for hot path optimization")
            except Exception as e:
                print(f"[WARN] CUDA warmup failed (non-fatal): {e}")
    except Exception as e:
        print(f"[ERROR] Failed to preload models: {e}")
        traceback.print_exc()
    finally:
        _warming_up = False
        print("[INFO] Background warmup finished")


@asynccontextmanager
async def lifespan(app):
    """应用生命周期管理：startup + shutdown

    P01 优化：GPU 激活同步执行（~10ms），模型预加载 + CUDA 预热
    作为后台任务运行，服务器立即开始接受连接。
    """
    # === Startup ===
    global _global_args, _cached_index_html, _warming_up
    # 如果通过 uvicorn 启动（非 __main__），_global_args 为 None，需要创建默认参数
    if _global_args is None:
        _global_args = argparse.Namespace(spk=True, language="中文", host="127.0.0.1", port=8000)
        print("[INFO] Created default _global_args (uvicorn launch)")
    print(f"[INFO] Startup event: _global_args.spk = {_global_args.spk}")

    # 缓存 index.html 内容
    html_path = os.path.join(WEB_DIR, "index.html")
    with open(html_path, encoding="utf-8") as f:
        _cached_index_html = f.read()
    print("[INFO] index.html cached")

    # 从配置文件加载持久化配置
    _app_config.load_from_file()

    # 激活 GPU P1 状态，避免空闲→活跃延迟（~10ms，保持同步）
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(0)
        _ = torch.zeros(1, device='cuda:0') + 1
        torch.cuda.synchronize()
        print("[INFO] GPU activated (P1 state)")
        _init_cuda_streams()

    # P01: 后台模型预加载 + CUDA 预热（非阻塞）
    if _global_args is not None and ASR_AVAILABLE:
        _warming_up = True
        asyncio.create_task(_background_model_warmup())

    yield  # 应用运行中

    # === Shutdown ===
    print("[INFO] Shutting down...")

    # 1. 清除模型缓存，释放模型引用
    try:
        clear_model_cache()
        print("[INFO] Model cache cleared")
    except Exception as e:
        print(f"[WARN] Failed to clear model cache: {e}")

    # 2. 等待所有 CUDA 操作完成
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            print("[INFO] CUDA synchronized")
        except Exception as e:
            print(f"[WARN] CUDA sync failed: {e}")

    # 3. 关闭线程池（所有 CUDA 工作已完成）
    print("[INFO] Shutting down thread pools...")
    _shutdown_executors()

    # 4. 恢复 empty_cache 并清理
    _enable_empty_cache()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARN] CUDA empty_cache failed: {e}")

    print("[INFO] Shutdown complete")


app = FastAPI(title="Realtime Transcription", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # P-10: 收紧 CORS，仅允许 localhost 常用端口（8000-9999）
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:[89]\d{3})?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get("/")
async def index():
    if _cached_index_html is not None:
        return HTMLResponse(content=_cached_index_html)
    html_path = os.path.join(WEB_DIR, "index.html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    models_loaded = _model_cache.stream_model is not None or _model_cache.offline_model is not None
    return {"status": "ok", "asr_available": ASR_AVAILABLE, "models_loaded": models_loaded, "warming_up": _warming_up}


def _build_device_status() -> dict:
    """构建设备/GPU 状态信息（前端 checkDeviceStatus() 依赖此结构）"""
    cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
    return {
        "current_setting": _model_cache.current_device,
        "resolved": _resolve_device(),
        "cuda_available": cuda_available,
        "torch_version": torch.__version__ if TORCH_AVAILABLE else "not installed",
        "cuda_version": torch.version.cuda if TORCH_AVAILABLE and torch.version.cuda else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "N/A",
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if cuda_available else "N/A",
    }


@app.get("/device/status")
async def device_status():
    """公开端点：返回设备/GPU 状态（前端 UI 用于显示 GPU 可用性）。

    不含敏感调试信息，所有用户均可访问。完整内部状态请用 /debug/status（需 --debug）。
    """
    return {"status": "ok", "device": _build_device_status()}


@app.get("/debug/status")
async def debug_status():
    """详细系统状态（供开发调试使用，--debug 参数启用）"""
    if not _DEBUG_MODE:
        return JSONResponse(status_code=404, content={"error": "Debug mode not enabled"})
    return {
        "status": "ok",
        "device": _build_device_status(),
        "models": {
            "stream_model_loaded": _model_cache.stream_model is not None,
            "vad_model_loaded": _model_cache.vad_model is not None,
            "offline_model_loaded": _model_cache.offline_model is not None,
            "offline_model_key": _model_cache.current_offline_model_key,
            "punc_model": "integrated in offline_model" if _model_cache.current_offline_model_key in ("seaco_paraformer", "funasr_nano", "funasr_nano_mlt") else "not applicable",
            "spk_model": "integrated in offline_model" if _model_cache.current_offline_model_key == "seaco_paraformer" and _model_cache.offline_model is not None else "not applicable",
        },
        "torch_available": TORCH_AVAILABLE,
        "asr_available": ASR_AVAILABLE,
    }


@app.get("/models")
async def get_models():
    """返回可用的 Pass2 离线模型列表"""
    return {
        "current": _model_cache.current_offline_model_key,
        "models": {k: {
            "name": v["name"],
            "desc": v["desc"],
            "latency_rank": v["latency_rank"],
            "latency_desc": v["latency_desc"],
            "latency_info": v["latency_info"],
            "supports_spk": v["supports_spk"],
            "supports_timestamps": v["supports_timestamps"],
            "supports_hotwords": v["supports_hotwords"],
            "lang_options": [{"value": lo[0], "label": lo[1]} for lo in v["lang_options"]],
            "available": _check_model_available(k)[0],
        } for k, v in OFFLINE_MODELS.items()},
    }


@app.get("/audio-devices")
async def get_audio_devices():
    """返回所有 WASAPI 环回设备（系统音频输出设备）"""
    if not WASAPI_AVAILABLE:
        return {"available": False, "devices": []}
    devices = WasapiAudioCapture.list_devices()
    return {"available": True, "devices": devices}


def _validate_bank_id(bank_id: str) -> bool:
    """校验说话人库 ID 是否合法（仅允许字母数字和下划线，防止路径遍历）"""
    if not bank_id or len(bank_id) > 64:
        return False
    if not all(c.isalnum() or c == '_' for c in bank_id):
        return False
    if bank_id.startswith('.') or '..' in bank_id:
        return False
    return True


def _safe_speaker_bank_path(bank_id: str) -> str | None:
    """安全获取说话人库文件路径，防止路径遍历攻击"""
    if not _validate_bank_id(bank_id):
        return None
    fpath = os.path.join(_SPEAKER_BANK_DIR, f"{bank_id}.json")
    real_path = os.path.realpath(fpath)
    real_base = os.path.realpath(_SPEAKER_BANK_DIR)
    if not real_path.startswith(real_base + os.sep) and real_path != real_base:
        return None
    return fpath


def _load_speaker_bank_from_file(fpath):
    """从 JSON 文件加载说话人库数据"""
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def _save_speaker_bank_to_file(bank_id: str, bank_name: str, speaker_data: dict) -> str:
    """将说话人库数据保存到文件，返回文件路径"""
    os.makedirs(_SPEAKER_BANK_DIR, exist_ok=True)
    fpath = os.path.join(_SPEAKER_BANK_DIR, f"{bank_id}.json")
    save_data = {
        "name": bank_name or bank_id,
        "centers": speaker_data["centers"],
        "counts": speaker_data["counts"],
        "max_speakers": speaker_data.get("max_speakers", 20),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False)
    return fpath


@app.get("/speaker-banks")
async def list_speaker_banks():
    """列出所有已保存的说话人库"""
    if not os.path.isdir(_SPEAKER_BANK_DIR):
        return {"banks": []}
    banks = []
    for fname in sorted(os.listdir(_SPEAKER_BANK_DIR)):
        if fname.endswith(".json"):
            fpath = os.path.join(_SPEAKER_BANK_DIR, fname)
            try:
                data = _load_speaker_bank_from_file(fpath)
                banks.append({
                    "id": fname[:-5],  # 去掉 .json
                    "name": data.get("name", fname[:-5]),
                    "num_speakers": len(data.get("centers", [])),
                    "created_at": data.get("created_at", ""),
                })
            except Exception as e:
                print(f"[WARN] Failed to read speaker bank file {fname}: {e}")
    return {"banks": banks}


@app.get("/speaker-bank/{bank_id}")
async def get_speaker_bank(bank_id: str):
    """加载指定说话人库"""
    fpath = _safe_speaker_bank_path(bank_id)
    if fpath is None:
        return JSONResponse(status_code=400, content={"error": "无效的 ID"})
    if not os.path.isfile(fpath):
        return JSONResponse(status_code=404, content={"error": f"说话人库 '{bank_id}' 不存在"})
    try:
        return _load_speaker_bank_from_file(fpath)
    except (json.JSONDecodeError, OSError) as e:
        # 文件解析错误：对用户有用，保留类别但避免泄露路径
        return JSONResponse(status_code=422, content={"error": "读取说话人库失败：文件损坏或不可读"})


@app.delete("/speaker-bank/{bank_id}")
async def delete_speaker_bank(bank_id: str):
    """删除指定说话人库"""
    fpath = _safe_speaker_bank_path(bank_id)
    if fpath is None:
        return JSONResponse(status_code=400, content={"error": "无效的 ID"})
    if os.path.isfile(fpath):
        try:
            os.remove(fpath)
            print(f"[INFO] Speaker bank deleted: {bank_id}")
            return {"ok": True}
        except OSError as e:
            print(f"[ERROR] Failed to delete speaker bank {bank_id}: {e}")
            return JSONResponse(status_code=500, content={"error": "删除失败，请查看服务器日志"})
    return JSONResponse(status_code=404, content={"error": f"说话人库 '{bank_id}' 不存在"})


@app.post("/speaker-bank/save-last")
async def save_last_speaker_bank(data: dict):
    """保存最后一次转录的说话人库数据（用于 WebSocket 断开后的保存）"""
    global _global_last_speaker_bank_data
    with _speaker_bank_global_lock:
        bank_data = _global_last_speaker_bank_data
    if not bank_data or not bank_data.get("centers"):
        return {"error": "没有可保存的说话人库数据（需要先进行转录并识别到说话人）"}

    bank_name = data.get("name", "").strip()
    bank_id = data.get("id", "").strip()
    fpath = _safe_speaker_bank_path(bank_id) if bank_id else None
    if fpath is None and bank_id:
        return {"error": "ID 只能包含字母、数字和下划线，且不能以点开头"}
    if not bank_id:
        bank_id = f"spk_{int(time.time())}"
    if not _validate_bank_id(bank_id):
        return {"error": "ID 只能包含字母、数字和下划线"}

    try:
        fpath = _save_speaker_bank_to_file(bank_id, bank_name, bank_data)
        print(f"[INFO] Speaker bank saved (last session): {bank_id} ({len(bank_data['centers'])} speakers)")
        return {"ok": True, "id": bank_id, "name": bank_name or bank_id, "num_speakers": len(bank_data["centers"])}
    except OSError as e:
        print(f"[ERROR] Failed to save speaker bank {bank_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "保存文件失败，请查看服务器日志"})


@app.post("/speaker-bank/load")
async def load_speaker_bank(data: dict):
    """通过 REST API 预加载说话人库（无需 WebSocket 连接），下次转录时自动应用"""
    global _preloaded_speaker_bank
    bank_id = data.get("id", "").strip()
    fpath = _safe_speaker_bank_path(bank_id)
    if fpath is None:
        return {"error": "无效的说话人库 ID"}
    if not os.path.isfile(fpath):
        return {"error": f"说话人库 '{bank_id}' 不存在"}
    try:
        bank_data = _load_speaker_bank_from_file(fpath)
        # 校验数据有效性（使用临时 SpeakerBank）
        tmp_bank = SpeakerBank()
        num_loaded = tmp_bank.load_from_dict(bank_data, config=_app_config)
        with _speaker_bank_global_lock:
            _preloaded_speaker_bank = bank_data
        print(f"[INFO] Speaker bank preloaded via REST: {bank_id} ({num_loaded} speakers)")
        return {
            "ok": True,
            "id": bank_id,
            "name": bank_data.get("name", bank_id),
            "num_speakers": num_loaded,
            "preset_spk_num": _app_config.preset_spk_num,
        }
    except ValueError as e:
        # 数据校验错误：对用户有用，保留
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:
        print(f"[ERROR] Failed to load speaker bank {bank_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "加载失败，请查看服务器日志"})


@app.get("/config")
async def get_config():
    return _app_config.to_dict()


@app.post("/config")
async def update_config(config: dict):
    try:
        async with _get_config_lock():
            _app_config.update(config)
            if "max_sentence_chars" in config:
                print(f"[INFO] Updated max_sentence_chars: {_app_config.max_sentence_chars}")
            if "vad_silence_gap_sec" in config:
                print(f"[INFO] Updated vad_silence_gap_sec: {_app_config.vad_silence_gap_sec}")
            if "vad_check_interval_sec" in config:
                print(f"[INFO] Updated vad_check_interval_sec: {_app_config.vad_check_interval_sec}")
            if "chunk_size_frames" in config:
                print(f"[INFO] Updated chunk_size_frames: {_app_config.chunk_size_frames} ({_app_config.chunk_size_frames * 60}ms)")
        return _app_config.to_dict()
    except Exception as e:
        print(f"[ERROR] Failed to update config: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": _INTERNAL_ERROR_MSG})


# ============================================================
# P-16: WebSocket 命令处理器类
# 将 13 个独立 _handle_* 函数组织为类方法，统一管理状态和依赖
# ============================================================

class CommandHandler:
    """WebSocket 命令处理器：封装所有命令处理逻辑"""

    # 精确匹配命令 -> 方法名
    EXACT_COMMANDS = {
        "START": "handle_start",
        "STOP": "handle_stop",
        "SPK_ON": "handle_spk_on",
        "SPK_OFF": "handle_spk_off",
        "CAPTURE_START": "handle_capture_start",
        "CAPTURE_STOP": "handle_capture_stop",
    }

    # 前缀匹配命令 -> 方法名
    PREFIX_COMMANDS = {
        "LANGUAGE:": "handle_language",
        "HOTWORDS:": "handle_hotwords",
        "SPK_SAVE:": "handle_spk_save",
        "SPK_LOAD:": "handle_spk_load",
        "MODEL_SWITCH:": "handle_model_switch",
        "DEVICE_SWITCH:": "handle_device_switch",
        "DEVICE_SELECT:": "handle_device_select",
    }

    def __init__(self, session, safe_send, client_info):
        self.session = session
        self.safe_send = safe_send
        self.client_info = client_info

    async def dispatch(self, msg: str) -> bool:
        """分发命令，返回是否已处理"""
        if msg in self.EXACT_COMMANDS:
            handler = getattr(self, self.EXACT_COMMANDS[msg])
            await handler()
            return True
        for prefix, method_name in self.PREFIX_COMMANDS.items():
            if msg.startswith(prefix):
                handler = getattr(self, method_name)
                await handler(msg[len(prefix):])
                return True
        return False

    async def handle_start(self):
        """处理 START 命令"""
        try:
            await self.session.start()
            print(f"[INFO] [{self.client_info}] Session started (2pass mode)")
            await self.safe_send({"event": "started"})
        except Exception as e:
            print(f"[ERROR] [{self.client_info}] Start failed: {e}")
            await self.safe_send({"event": "error", "message": str(e)})

    async def handle_stop(self):
        """处理 STOP 命令"""
        result = await self.session.stop()
        print(f"[INFO] [{self.client_info}] Session stopped, sentences={len(self.session.sentences)}")
        await self.safe_send(result)

    async def handle_language(self, msg: str):
        """处理 LANGUAGE 命令"""
        lang = msg.strip()
        if not lang:
            await self.safe_send({"event": "error", "message": "语言不能为空"})
            return
        current_model_key = _model_cache.current_offline_model_key
        model_info = OFFLINE_MODELS.get(current_model_key, {})
        lang_options = model_info.get("lang_options", [])
        supported_langs = [opt[0] for opt in lang_options] if lang_options else []
        if supported_langs and lang != "auto" and lang not in supported_langs:
            print(f"[WARN] [{self.client_info}] Language {lang} may not be supported by model {current_model_key}")
        self.session.set_language(lang)
        print(f"[INFO] [{self.client_info}] Language set: {lang}")
        await self.safe_send({"event": "language_set", "language": lang})

    async def handle_hotwords(self, msg: str):
        """处理 HOTWORDS 命令"""
        hw = msg.strip()
        current_model_key = _model_cache.current_offline_model_key
        model_info = OFFLINE_MODELS.get(current_model_key, {})
        if not model_info.get("supports_hotwords", False):
            await self.safe_send({"event": "error", "message": f"当前模型 {model_info.get('name', current_model_key)} 不支持热词功能"})
            print(f"[WARN] [{self.client_info}] HOTWORDS rejected: model {current_model_key} does not support hotwords")
            return
        if len(hw) > 2000:
            await self.safe_send({"event": "error", "message": "热词内容过长，最多 2000 字符"})
            return
        words = [w.strip() for w in hw.replace('\n', ',').replace('，', ',').split(',') if w.strip()]
        if len(words) > 100:
            await self.safe_send({"event": "error", "message": f"热词数量过多（{len(words)}个），最多 100 个"})
            return
        self.session.set_hotwords(hw)
        print(f"[INFO] [{self.client_info}] Hotwords set: {hw[:50]}")
        await self.safe_send({"event": "hotwords_set"})

    async def handle_spk_on(self):
        """处理 SPK_ON 命令"""
        current_model_key = _model_cache.current_offline_model_key
        model_info = OFFLINE_MODELS.get(current_model_key, {})
        if not model_info.get("supports_spk", False):
            await self.safe_send({"event": "spk_error", "message": f"当前模型 {model_info.get('name', current_model_key)} 不支持说话人分离，请切换到 SeACoParaformer 模型"})
            print(f"[WARN] [{self.client_info}] SPK_ON rejected: model {current_model_key} does not support spk")
            return
        async with _get_model_lock():
            _global_args.spk = True
        print(f"[INFO] [{self.client_info}] Speaker diarization ON")
        if self.session.is_active:
            try:
                await self.session.reload_models()
                print(f"[INFO] [{self.client_info}] Offline model reloaded with speaker diarization")
            except Exception as e:
                print(f"[ERROR] [{self.client_info}] Failed to reload model with spk: {e}")
        await self.safe_send({"event": "spk_on"})

    async def handle_spk_off(self):
        """处理 SPK_OFF 命令"""
        async with _get_model_lock():
            _global_args.spk = False
        print(f"[INFO] [{self.client_info}] Speaker diarization OFF")
        if self.session.is_active:
            try:
                await self.session.reload_models()
                print(f"[INFO] [{self.client_info}] Offline model reloaded without speaker diarization")
            except Exception as e:
                print(f"[ERROR] [{self.client_info}] Failed to reload model without spk: {e}")
        await self.safe_send({"event": "spk_off"})

    async def handle_spk_save(self, msg: str):
        """处理 SPK_SAVE 命令：保存当前说话人库"""
        try:
            data = json.loads(msg) if msg else {}
        except Exception:
            data = {}
        bank_name = data.get("name", "")
        bank_id = data.get("id", "")

        speaker_data = self.session.speaker_bank.save_to_dict()
        if not speaker_data["centers"]:
            await self.safe_send({"event": "spk_save_error", "message": "当前没有说话人数据可保存（需要先识别到说话人）"})
            return

        if not bank_id:
            bank_id = f"spk_{int(time.time())}"

        try:
            _save_speaker_bank_to_file(bank_id, bank_name, speaker_data)
        except OSError as e:
            await self.safe_send({"event": "spk_save_error", "message": f"保存文件失败: {e}"})
            return

        print(f"[INFO] [{self.client_info}] Speaker bank saved: {bank_id} ({len(speaker_data['centers'])} speakers)")
        await self.safe_send({
            "event": "spk_saved",
            "id": bank_id,
            "name": bank_name or bank_id,
            "num_speakers": len(speaker_data["centers"]),
        })

    async def handle_spk_load(self, msg: str):
        """处理 SPK_LOAD 命令：加载说话人库"""
        bank_id = msg.strip()
        fpath = _safe_speaker_bank_path(bank_id)
        if fpath is None:
            await self.safe_send({"event": "spk_load_error", "message": "无效的说话人库 ID"})
            return
        if not os.path.isfile(fpath):
            await self.safe_send({"event": "spk_load_error", "message": f"说话人库 '{bank_id}' 不存在"})
            return
        try:
            data = _load_speaker_bank_from_file(fpath)
            num_loaded = self.session.speaker_bank.load_from_dict(data, config=_app_config)
            print(f"[INFO] [{self.client_info}] Speaker bank loaded: {bank_id} ({num_loaded} speakers), preset_spk_num synced to {num_loaded}")
            await self.safe_send({
                "event": "spk_loaded",
                "id": bank_id,
                "name": data.get("name", bank_id),
                "num_speakers": num_loaded,
                "preset_spk_num": _app_config.preset_spk_num,
            })
        except Exception as e:
            print(f"[ERROR] [{self.client_info}] Failed to load speaker bank: {e}")
            await self.safe_send({"event": "spk_load_error", "message": "加载失败，请查看服务器日志"})

    async def handle_model_switch(self, msg: str):
        """处理 MODEL_SWITCH 命令"""
        model_key = msg.strip()
        if model_key not in OFFLINE_MODELS:
            print(f"[WARN] [{self.client_info}] Unknown model key: {model_key}")
            await self.safe_send({"event": "model_error", "message": f"Unknown model: {model_key}"})
            return
        available, reason = _check_model_available(model_key)
        if not available:
            print(f"[WARN] [{self.client_info}] Model not available locally: {reason}")
            await self.safe_send({"event": "model_error", "message": f"模型未下载到本地: {OFFLINE_MODELS[model_key]['name']}"})
            return
        async with _get_model_lock():
            old_key = _model_cache.current_offline_model_key
            _model_cache.current_offline_model_key = model_key
        print(f"[INFO] [{self.client_info}] Switching offline model: {old_key} → {model_key}")
        model_info = OFFLINE_MODELS[model_key]

        warnings = []
        if not model_info.get("supports_spk") and _global_args.spk:
            async with _get_model_lock():
                _global_args.spk = False
            warnings.append("新模型不支持说话人分离，已自动关闭")
            print(f"[INFO] [{self.client_info}] Auto-disabled speaker diarization (model does not support spk)")

        if self.session.is_active:
            try:
                await self.session.reload_models()
                response = {
                    "event": "model_switched",
                    "model": model_key,
                    "name": model_info["name"],
                    "desc": model_info["desc"],
                    "supports_spk": model_info["supports_spk"],
                    "supports_timestamps": model_info["supports_timestamps"],
                }
                if warnings:
                    response["warnings"] = warnings
                await self.safe_send(response)
                print(f"[INFO] [{self.client_info}] Model switched to {model_key}")
            except Exception as e:
                print(f"[ERROR] [{self.client_info}] Model switch failed: {e}")
                traceback.print_exc()
                async with _get_model_lock():
                    _model_cache.current_offline_model_key = old_key
                await self.safe_send({"event": "model_error", "message": f"Failed to load model {model_key}: {e}"})
        else:
            response = {
                "event": "model_switched",
                "model": model_key,
                "name": model_info["name"],
                "desc": model_info["desc"],
                "supports_spk": model_info["supports_spk"],
                "supports_timestamps": model_info["supports_timestamps"],
            }
            if warnings:
                response["warnings"] = warnings
            await self.safe_send(response)

    async def handle_device_switch(self, msg: str):
        """处理 DEVICE_SWITCH 命令"""
        new_device = msg.strip()
        if new_device not in ("auto", "cpu", "cuda"):
            await self.safe_send({"event": "device_error", "message": f"Unknown device: {new_device}"})
            return
        async with _get_model_lock():
            old_device = _model_cache.current_device
            _model_cache.current_device = new_device
        print(f"[INFO] [{self.client_info}] Switching device: {old_device} → {new_device}")
        resolved = _resolve_device()

        if self.session.is_active:
            try:
                await self.session.reload_models()
                await self.safe_send({"event": "device_switched", "resolved": resolved})
                print(f"[INFO] [{self.client_info}] Device switched to {resolved}")
            except Exception as e:
                print(f"[ERROR] [{self.client_info}] Device switch failed: {e}")
                traceback.print_exc()
                async with _get_model_lock():
                    _model_cache.current_device = old_device
                await self.safe_send({"event": "device_error", "message": f"Failed to switch device: {e}"})
        else:
            await self.safe_send({"event": "device_switched", "resolved": resolved})

    async def handle_capture_start(self, msg: str = ""):
        """处理 CAPTURE_START 命令"""
        try:
            device_index = _model_cache.current_audio_device_index
            await self.session.start_capture(device_index=device_index)
            await self.safe_send({"event": "capture_started"})
            print(f"[INFO] [{self.client_info}] Audio capture started (device_index={device_index})")
        except Exception as e:
            print(f"[ERROR] [{self.client_info}] Capture start failed: {e}")
            traceback.print_exc()
            await self.safe_send({"event": "capture_error", "message": str(e)})

    async def handle_capture_stop(self, msg: str = ""):
        """处理 CAPTURE_STOP 命令"""
        try:
            await self.session.stop_capture()
            await self.safe_send({"event": "capture_stopped"})
            print(f"[INFO] [{self.client_info}] Audio capture stopped")
        except Exception as e:
            print(f"[ERROR] [{self.client_info}] Capture stop failed: {e}")
            await self.safe_send({"event": "capture_error", "message": str(e)})

    async def handle_device_select(self, msg: str):
        """处理 DEVICE_SELECT 命令"""
        dev_idx_str = msg.strip()
        if dev_idx_str == "auto":
            async with _get_model_lock():
                _model_cache.current_audio_device_index = None
            await self.safe_send({"event": "device_selected", "device_index": None})
        else:
            try:
                dev_idx = int(dev_idx_str)
                if dev_idx < 0:
                    await self.safe_send({"event": "device_select_error", "message": f"Invalid device index: {dev_idx_str} (must be non-negative)"})
                    return
                if dev_idx > 100:
                    await self.safe_send({"event": "device_select_error", "message": f"Invalid device index: {dev_idx_str} (out of range)"})
                    return
                async with _get_model_lock():
                    _model_cache.current_audio_device_index = dev_idx
                await self.safe_send({"event": "device_selected", "device_index": dev_idx})
            except ValueError:
                await self.safe_send({"event": "device_select_error", "message": f"Invalid device index: {dev_idx_str}"})


@app.websocket("/ws")
async def websocket_handler(client_ws: WebSocket):
    await client_ws.accept()
    client_info = f"{client_ws.client.host}:{client_ws.client.port}"
    print(f"[INFO] Client connected: {client_info}")

    session = AsrSession(args=_global_args)
    session.websocket = client_ws

    # 如果用户在转录前通过 REST API 预加载了说话人库，自动应用到 session
    preloaded_data = None
    with _speaker_bank_global_lock:
        preloaded_data = _preloaded_speaker_bank
    if preloaded_data is not None:
        try:
            num = session.speaker_bank.load_from_dict(preloaded_data, config=_app_config)
            print(f"[INFO] [{client_info}] Preloaded speaker bank applied ({num} speakers)")
        except Exception as e:
            print(f"[WARN] [{client_info}] Failed to apply preloaded speaker bank: {e}")

    async def safe_send(data):
        """安全发送 JSON，客户端已断开时返回 False"""
        try:
            await client_ws.send_json(data)
            return True
        except Exception:
            return False

    cmd_handler = CommandHandler(session, safe_send, client_info)

    try:
        while True:
            data = await client_ws.receive()
            if data.get("type") == "websocket.disconnect":
                break
            if "text" in data:
                msg = data["text"]
                if not msg.startswith('AUDIO:'):
                    print(f"[INFO] [{client_info}] Command: {msg}")

                # P-16: 使用 CommandHandler 类分发命令
                handled = await cmd_handler.dispatch(msg)

                if not handled and msg.startswith("AUDIO:"):
                    # 音频数据（base64 编码）
                    audio_bytes = base64.b64decode(msg[6:])
                    if len(audio_bytes) > _MAX_FRAME_BYTES:
                        print(f"[WARN] [{client_info}] Audio frame too large: {len(audio_bytes)} bytes, max {_MAX_FRAME_BYTES}")
                        continue
                    await session.process_audio(audio_bytes)
                elif not handled:
                    # 未识别的文本命令，发送错误反馈
                    await safe_send({"event": "error", "message": f"未知命令: {msg[:50]}"})

            elif "bytes" in data:
                await session.process_audio(data["bytes"])

    except WebSocketDisconnect:
        print(f"[INFO] Client disconnected: {client_info}")
    except Exception as e:
        print(f"[ERROR] WebSocket error [{client_info}]: {e}")
        traceback.print_exc()
    finally:
        # 保存说话人库数据到全局暂存
        with _speaker_bank_global_lock:
            if session._last_speaker_bank_data:
                _global_last_speaker_bank_data = session._last_speaker_bank_data
            elif session.speaker_bank and len(session.speaker_bank.centers) > 0:
                try:
                    _global_last_speaker_bank_data = session.speaker_bank.save_to_dict()
                except Exception as e:
                    print(f"[WARN] Failed to save speaker bank on disconnect: {e}")
        await session.cleanup()
        print(f"[INFO] Session cleaned up: {client_info}")


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime Transcription Server (2pass Architecture)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--language", type=str, default="中文")
    parser.add_argument("--spk", action="store_true", dest="spk", help="Enable speaker diarization")
    parser.add_argument("--no-spk", action="store_false", dest="spk", help="Disable speaker diarization")
    parser.add_argument("--debug", action="store_true", dest="debug", help="Enable debug endpoints")
    parser.set_defaults(spk=True)  # 默认启用说话人分离
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _global_args = args
    _DEBUG_MODE = args.debug

    if _DEBUG_MODE:
        print("[INFO] Debug mode enabled — /debug/status endpoint available")
    print(f"[INFO] Starting server with args: spk={args.spk}, host={args.host}, port={args.port}")
    print(f"[INFO] Speaker diarization: {'ON' if args.spk else 'OFF'}")
    print(f"[INFO] Architecture: 2pass (Pass1=streaming, Pass2=offline+VAD+punc+spk)")
    print(f"[INFO] Streaming chunk: {_app_config.chunk_size_frames * 60}ms ({_app_config.chunk_size_frames} frames)")
    print(f"[INFO] VAD check interval: {_app_config.vad_check_interval_sec}s, "
          f"silence gap: {_app_config.vad_silence_gap_sec}s")
    print(f"[INFO] Max sentence chars: {_app_config.max_sentence_chars}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
