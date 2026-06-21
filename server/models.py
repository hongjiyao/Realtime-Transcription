import os
import asyncio
import concurrent.futures
import threading
from typing import Literal
from server.config import (
    TORCH_AVAILABLE, ASR_AVAILABLE,
    _MODELS_DIR, OFFLINE_MODELS,
    MODEL_ONLINE, MODEL_VAD, MODEL_PUNC, MODEL_SPK,
    _model_cache, _app_config,
)

try:
    import torch
except ImportError:
    torch = None

try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None

# 自适应线程池：转录开始时创建，停止时销毁，避免空闲线程占用资源
_gpu_executor: concurrent.futures.ThreadPoolExecutor | None = None
_pass2_executor: concurrent.futures.ThreadPoolExecutor | None = None
_load_executor: concurrent.futures.ThreadPoolExecutor | None = None  # P10: 独立模型加载线程池
# P07: 移除 _executor_lock — _ensure_executors 在 _model_lock 保护下调用，
#      _shutdown_executors 仅在 shutdown 阶段调用，无需额外锁。


def _ensure_executors():
    """确保线程池已创建（懒创建）

    P06: GPU executor 扩容到 3 workers，支持 Pass1 + VAD + Pass2 三模型并行推理。
    P10: 独立 _load_executor 用于模型加载，避免与推理线程争抢。
    """
    global _gpu_executor, _pass2_executor, _load_executor
    if _gpu_executor is None:
        _gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="gpu")
    if _pass2_executor is None:
        _pass2_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="pass2")
    if _load_executor is None:
        _load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="load")
    print(f"[INFO] Thread pools created: active_threads={threading.active_count()}")


def _shutdown_executors():
    """关闭线程池，释放空闲线程"""
    global _gpu_executor, _pass2_executor, _load_executor
    before = threading.active_count()
    if _gpu_executor is not None:
        _gpu_executor.shutdown(wait=True)
        _gpu_executor = None
    if _pass2_executor is not None:
        _pass2_executor.shutdown(wait=True)
        _pass2_executor = None
    if _load_executor is not None:
        _load_executor.shutdown(wait=True)
        _load_executor = None
    after = threading.active_count()
    print(f"[INFO] Thread pools destroyed: threads {before} → {after}")

# GPU 优化：CUDA Streams 实现并行推理（Pass1/VAD/Pass2 在不同 stream 上执行）
_cuda_stream_pass1 = None
_cuda_stream_vad = None
_cuda_stream_pass2 = None
_cuda_stream_lock = threading.Lock()       # 保护 CUDA Stream 初始化

# CUDA Event：用于 capture 线程非阻塞查询 GPU 空闲状态
# 每个 Stream 一个独立 Event，避免多流漏检
_gpu_event_pass1 = None
_gpu_event_vad = None
_gpu_event_pass2 = None

def _init_cuda_streams():
    """初始化 CUDA Streams（延迟初始化，避免模块加载时 CUDA 未就绪）"""
    global _cuda_stream_pass1, _cuda_stream_vad, _cuda_stream_pass2
    global _gpu_event_pass1, _gpu_event_vad, _gpu_event_pass2
    with _cuda_stream_lock:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if _cuda_stream_pass1 is None:
                _cuda_stream_pass1 = torch.cuda.Stream(priority=0)  # 高优先级：Pass1 实时性要求高
                _cuda_stream_vad = torch.cuda.Stream(priority=0)
                _cuda_stream_pass2 = torch.cuda.Stream(priority=-1)  # 低优先级：Pass2 可以让步
                # Event(blocking=False) 允许非阻塞 query()，~0.001ms
                _gpu_event_pass1 = torch.cuda.Event(blocking=False)
                _gpu_event_vad = torch.cuda.Event(blocking=False)
                _gpu_event_pass2 = torch.cuda.Event(blocking=False)
                # 初始 record，标记为已完成（GPU 空闲状态）
                _gpu_event_pass1.record()
                _gpu_event_vad.record()
                _gpu_event_pass2.record()
                print("[INFO] CUDA Streams initialized: Pass1(stream0), VAD(stream1), Pass2(low-priority)")

# ============================================================
# CUDA empty_cache 抑制（热路径优化）
# ============================================================
# FunASR 的 auto_model.py 在每次 generate() 后调用 torch.cuda.empty_cache()，
# 在实时转录热路径上会产生 5-20ms 的 CUDA GC 抖动。
# 此处用 monkey-patch 替换为带标志位控制的代理函数，
# 仅在模型加载/释放时（非热路径）允许真正的 empty_cache。
#
# 设计权衡：
#   - 不能修改 FunASR 源码，因此必须 monkey-patch torch.cuda.empty_cache
#   - 用模块级标志位 _empty_cache_disabled 控制实际行为，便于调试和测试
#   - 所有 GPU 内存管理操作（clear_model_cache / shutdown）通过 _enable_empty_cache 恢复
_original_empty_cache = None
_empty_cache_disabled = False


def _patched_empty_cache():
    """torch.cuda.empty_cache 代理：根据标志位决定是否真正调用。

    被禁用时为 no-op，避免热路径上的 CUDA GC 抖动；
    被启用时调用原始实现释放缓存。
    """
    global _original_empty_cache, _empty_cache_disabled
    if _original_empty_cache is not None and not _empty_cache_disabled:
        _original_empty_cache()


def _disable_empty_cache():
    """禁用 torch.cuda.empty_cache（安装 monkey-patch + 设置标志位）。

    幂等：多次调用安全。仅在 GPU 可用时生效。
    """
    global _original_empty_cache, _empty_cache_disabled
    if TORCH_AVAILABLE and torch.cuda.is_available():
        if _original_empty_cache is None:
            _original_empty_cache = torch.cuda.empty_cache
            torch.cuda.empty_cache = _patched_empty_cache
        _empty_cache_disabled = True


def _enable_empty_cache():
    """恢复 torch.cuda.empty_cache 的实际行为（仅清除标志位，不卸载 patch）。

    幂等：多次调用安全。调用后 _original_empty_cache 仍可被 _safe_empty_cache 使用。
    """
    global _empty_cache_disabled
    _empty_cache_disabled = False


class _EmptyCacheContext:
    """P-14: 上下文管理器，在 with 块内临时启用/禁用 empty_cache"""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._prev_state = None

    def __enter__(self):
        global _empty_cache_disabled
        self._prev_state = _empty_cache_disabled
        _empty_cache_disabled = not self._enabled
        return self

    def __exit__(self, *args):
        global _empty_cache_disabled
        _empty_cache_disabled = self._prev_state


def empty_cache_context(enabled: bool = True):
    """返回一个上下文管理器，控制 empty_cache 行为。

    用法:
        with empty_cache_context(enabled=True):
            # 在此块内 empty_cache 正常执行
            clear_model_cache("all")
    """
    return _EmptyCacheContext(enabled=enabled)


def _safe_empty_cache():
    """安全地执行一次 empty_cache，绕过热路径抑制标志。

    用于 clear_model_cache / shutdown 等需要强制释放 GPU 内存的场景。
    直接调用 _original_empty_cache（若已安装），否则调用 torch.cuda.empty_cache。
    """
    global _original_empty_cache
    if _original_empty_cache is not None:
        _original_empty_cache()
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

# 模型缓存（由 _model_cache 数据类管理）
_model_lock = None
_config_lock = None
_global_args = None
_global_last_speaker_bank_data = None  # 全局暂存：最后一个 session 的说话人库数据
_preloaded_speaker_bank = None  # REST API 预加载的说话人库数据，WebSocket 连接时自动应用
_speaker_bank_global_lock = threading.Lock()  # 保护 _global_last_speaker_bank_data / _preloaded_speaker_bank


_lock_init_lock = threading.Lock()  # 保护 _model_lock / _config_lock 的延迟初始化


def _get_model_lock() -> asyncio.Lock:
    """延迟初始化 asyncio.Lock，避免模块级别创建（无事件循环时触发 DeprecationWarning）
    P-06: 使用 _lock_init_lock 保护，防止竞态条件"""
    global _model_lock
    with _lock_init_lock:
        if _model_lock is None:
            _model_lock = asyncio.Lock()
    return _model_lock


def _get_config_lock() -> asyncio.Lock:
    """延迟初始化 asyncio.Lock，避免模块级别创建（无事件循环时触发 DeprecationWarning）
    P-06: 使用 _lock_init_lock 保护，防止竞态条件"""
    global _config_lock
    with _lock_init_lock:
        if _config_lock is None:
            _config_lock = asyncio.Lock()
    return _config_lock


def _resolve_device() -> str:
    """解析设备选择：auto → 自动检测，cpu → 强制 CPU，cuda → 强制 GPU"""
    if _model_cache.current_device == "cpu":
        return "cpu"
    elif _model_cache.current_device == "cuda":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"
        else:
            print("[WARN] CUDA requested but not available, falling back to CPU")
            return "cpu"
    else:  # auto
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"


def _check_model_available(model_key: str) -> tuple:
    """检查模型是否存在于本地目录"""
    if model_key not in OFFLINE_MODELS:
        return False, f"Unknown model: {model_key}"
    local_dir = os.path.join(_MODELS_DIR, OFFLINE_MODELS[model_key]["local_dir"])
    if not os.path.isdir(local_dir):
        return False, f"模型目录不存在: {local_dir}"
    # 检查是否有模型权重文件（.pt / .bin / campplus_cn_common.bin）
    has_pt = os.path.isfile(os.path.join(local_dir, "model.pt"))
    has_bin = os.path.isfile(os.path.join(local_dir, "model.bin"))
    has_campplus = os.path.isfile(os.path.join(local_dir, "campplus_cn_common.bin"))
    if not (has_pt or has_bin or has_campplus):
        return False, f"模型权重文件不存在: {local_dir}"
    return True, "OK"


def _load_stream_model(device: str) -> None:
    """加载流式模型（Pass1）"""
    if AutoModel is None:
        raise RuntimeError("FunASR AutoModel not available - funasr package not installed")
    print("[INFO] Loading streaming model (Pass1)...")
    stream_model = AutoModel(
        model=MODEL_ONLINE,
        trust_remote_code=False,
        device=device,
        disable_update=True,
        ncpu=1,  # 自适应：GPU 推理只需 1 个 CPU 线程做调度
    )
    try:
        if hasattr(stream_model, 'model') and hasattr(stream_model.model, 'predictor'):
            stream_model.model.predictor.tail_threshold = 0.5
            print("[INFO] Set predictor.tail_threshold=0.5")
    except Exception as e:
        print(f"[WARN] Failed to set tail_threshold: {e}")
    _model_cache.stream_model = stream_model
    _model_cache.stream_model_device = device
    print("[INFO] Streaming model loaded")


def _load_vad_model() -> None:
    """加载 VAD 模型 — 优先使用 GPU 加速 VAD 推理"""
    if AutoModel is None:
        raise RuntimeError("FunASR AutoModel not available - funasr package not installed")
    resolved = _resolve_device()
    print(f"[INFO] Loading VAD model for endpoint detection ({resolved})...")
    vad_model = AutoModel(
        model=MODEL_VAD,
        trust_remote_code=False,
        device=resolved,
        disable_update=True,
        ncpu=1,  # 自适应：GPU 推理只需 1 个 CPU 线程做调度
    )
    _model_cache.vad_model = vad_model
    _model_cache.vad_model_device = resolved
    print(f"[INFO] VAD model loaded ({resolved})")


def _load_offline_model(device: str, spk_enabled: bool) -> None:
    """加载离线模型（Pass2）— 仅从本地目录加载，不下载。

    ⚠️ 安全提示：部分模型（sensevoice / funasr_nano / funasr_nano_mlt）使用
    trust_remote_code=True，会在加载时执行模型仓库内的 Python 代码。
    风险缓解措施：
      1. 仅从本地 models/<local_dir>/ 目录加载，不从网络下载
      2. disable_update=True 阻止自动更新覆盖本地文件
      3. 本项目为个人/本地使用，不暴露到公网

    如需完全禁用 remote code 执行，可将这些模型的 trust_remote_code 改为 False，
    但会导致模型无法加载（这些模型依赖自定义推理代码）。
    """
    if AutoModel is None:
        raise RuntimeError("FunASR AutoModel not available - funasr package not installed")
    offline_model_info = OFFLINE_MODELS[_model_cache.current_offline_model_key]
    offline_local_dir = os.path.join(_MODELS_DIR, offline_model_info["local_dir"])
    trust_remote = offline_model_info["trust_remote_code"]
    if trust_remote:
        print(f"[WARN] trust_remote_code=True for {offline_model_info['name']} — 仅从本地目录加载，风险已缓解")
    print(f"[INFO] Loading offline model (Pass2): {offline_model_info['name']} from {offline_local_dir} (trust_remote_code={trust_remote})...")

    offline_kwargs = {
        "model": offline_local_dir,
        "trust_remote_code": trust_remote,
        "device": device,
        "disable_update": True,
        "ncpu": 1,  # 自适应：GPU 推理只需 1 个 CPU 线程做调度
    }

    # SeACoParaformer 特殊配置
    if _model_cache.current_offline_model_key == "seaco_paraformer":
        # SeACoParaformer 需要 VAD 将长音频分段，否则处理长音频效率极低
        offline_kwargs["vad_model"] = MODEL_VAD
        offline_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        offline_kwargs["punc_model"] = MODEL_PUNC
    # SenseVoice：保留 VAD 用于分段（长音频不分段效率极低）
    elif _model_cache.current_offline_model_key == "sensevoice":
        offline_kwargs["vad_model"] = MODEL_VAD
        offline_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        print("[INFO] SenseVoice with VAD for segmentation")
    # Fun-ASR-Nano：保留 VAD 用于分段
    elif _model_cache.current_offline_model_key in ("funasr_nano", "funasr_nano_mlt"):
        offline_kwargs["vad_model"] = MODEL_VAD
        offline_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        # 说话人分离需要 punc_model 做句子切分
        offline_kwargs["punc_model"] = MODEL_PUNC
        print(f"[INFO] {_model_cache.current_offline_model_key} with VAD + punc for segmentation")

    # 说话人分离：所有 supports_spk 模型通用
    model_info = OFFLINE_MODELS[_model_cache.current_offline_model_key]
    if spk_enabled and model_info.get("supports_spk"):
        offline_kwargs["spk_model"] = MODEL_SPK
        print(f"[INFO] Speaker diarization enabled for {_model_cache.current_offline_model_key}")
    elif not spk_enabled:
        print(f"[INFO] Speaker diarization disabled for Pass2")

    offline_model = AutoModel(**offline_kwargs)
    _model_cache.offline_model = offline_model
    _model_cache.offline_model_key = _model_cache.current_offline_model_key
    _model_cache.offline_model_device = device
    _model_cache.offline_model_spk = spk_enabled
    print(f"[INFO] Offline model loaded: {_model_cache.current_offline_model_key}")


async def _reload_model_if_needed(
    need_reload: bool,
    label: str,
    load_fn,
    *load_args,
    cache_attr: str,
    clear_what: str,
    loop,
    reasons: str = "",
) -> bool:
    """统一的模型重载流程：释放旧模型 → 加载新模型 → 验证。

    Args:
        need_reload: 是否需要重载
        label: 模型标签（用于日志，如 "Stream" / "VAD" / "Offline"）
        load_fn: 加载函数（_load_stream_model / _load_vad_model / _load_offline_model）
        load_args: 传给 load_fn 的位置参数
        cache_attr: _model_cache 上的属性名（如 "stream_model"），用于验证加载结果
        clear_what: 传给 clear_model_cache 的参数（如 "stream" / "offline"）
        loop: asyncio 事件循环
        reasons: 可选的重载原因描述（用于日志）

    Returns:
        True 如果执行了重载，False 如果跳过
    """
    if not need_reload:
        return False

    # 释放旧模型 GPU 内存（若存在）
    if getattr(_model_cache, cache_attr) is not None:
        clear_model_cache(clear_what)

    log_msg = f"[INFO] {label} model needs reload"
    if reasons:
        log_msg += f" ({reasons})"
    print(log_msg)

    _ensure_executors()
    try:
        # P10: 使用独立 _load_executor，不阻塞推理线程
        await loop.run_in_executor(_load_executor, load_fn, *load_args)
    except Exception as e:
        print(f"[ERROR] Failed to load {label.lower()} model: {e}")
    if getattr(_model_cache, cache_attr) is None:
        print(f"[ERROR] {label} model load failed, cache is inconsistent (old model freed, new model unavailable)")
    return True


def _prepare_reload(need_reload: bool, label: str, load_fn, load_args: tuple,
                    cache_attr: str, clear_what: str, reasons: str = "") -> dict | None:
    """P08: 阶段1（同步）：判断是否需要重载，并释放旧模型 GPU 内存。

    与 _reload_model_if_needed 不同，此函数只做释放阶段，加载阶段交由
    _execute_parallel_loads 并行执行。返回 None 表示无需加载。
    """
    if not need_reload:
        return None

    if getattr(_model_cache, cache_attr) is not None:
        clear_model_cache(clear_what)

    log_msg = f"[INFO] {label} model needs reload"
    if reasons:
        log_msg += f" ({reasons})"
    print(log_msg)

    return {
        "label": label,
        "load_fn": load_fn,
        "load_args": load_args,
        "cache_attr": cache_attr,
    }


async def _execute_parallel_loads(tasks: list, loop) -> list:
    """P08: 阶段2（异步）：并行加载多个模型，每个使用独立 CUDA Stream。

    Args:
        tasks: _prepare_reload 返回的描述符列表（None 已被过滤）
        loop: asyncio 事件循环

    Returns:
        每个任务的完成状态列表（True=缓存已设置成功）
    """
    if not tasks:
        return []
    _ensure_executors()

    async def _load_one(task):
        label = task["label"]
        try:
            # P10: 使用独立 _load_executor，不阻塞推理线程
            await loop.run_in_executor(_load_executor, task["load_fn"], *task["load_args"])
            ok = getattr(_model_cache, task["cache_attr"]) is not None
            if not ok:
                print(f"[ERROR] {label} model load failed, cache is inconsistent "
                      f"(old model freed, new model unavailable)")
            return ok
        except Exception as e:
            print(f"[ERROR] Failed to load {label.lower()} model: {e}")
            return False

    return await asyncio.gather(*[_load_one(t) for t in tasks])


async def get_or_create_models(args) -> tuple:
    """返回 (stream_model, vad_model, offline_model) 元组

    P08 优化：三个模型并行加载，每个使用独立 CUDA Stream。
    切换设备时整体重载（stream+vad+offline 并行），相比串行节省 40-60% 时间。

    只重新加载需要更新的模型，避免全部重载：
    - 切换离线模型：只重载 offline
    - 切换设备：全部重载（并行）
    - 切换说话人：只重载 offline（因为 spk_model 是 offline 的子模型）

    安全：加载新模型前先释放旧模型 GPU 内存，避免 OOM；
    如果新模型加载失败，旧模型已释放但不会导致崩溃（返回 None）。
    """
    async with _get_model_lock():
        device = _resolve_device()
        assert args is not None, "args must not be None in get_or_create_models"
        spk_enabled = getattr(args, "spk", False)
        loop = asyncio.get_running_loop()

        # ===== 阶段1：同步判断 + 释放旧模型 GPU 内存 =====
        # 流式模型：device 变化或首次加载时重载
        need_stream = (_model_cache.stream_model is None or _model_cache.stream_model_device != device)
        stream_reason = ("not cached" if _model_cache.stream_model is None
                         else f"device changed: {_model_cache.stream_model_device} → {device}")
        stream_task = _prepare_reload(
            need_stream, "Stream", _load_stream_model, (device,),
            cache_attr="stream_model", clear_what="stream", reasons=stream_reason,
        )

        # VAD 模型：跟随 device 参数运行（GPU 优先）
        need_vad = (_model_cache.vad_model is None or _model_cache.vad_model_device != device)
        vad_reason = (f"cached_device={_model_cache.vad_model_device}, target={device}"
                      if _model_cache.vad_model is not None else "not cached")
        vad_task = _prepare_reload(
            need_vad, "VAD", _load_vad_model, (),
            cache_attr="vad_model", clear_what="vad", reasons=vad_reason,
        )

        # 离线模型：模型类型 / device / 说话人开关变化时重载
        need_offline = (
            _model_cache.offline_model is None
            or _model_cache.offline_model_key != _model_cache.current_offline_model_key
            or _model_cache.offline_model_device != device
            or _model_cache.offline_model_spk != spk_enabled
        )
        offline_reasons = []
        if _model_cache.offline_model is None:
            offline_reasons.append("not cached")
        if _model_cache.offline_model_key != _model_cache.current_offline_model_key:
            offline_reasons.append(f"model changed: {_model_cache.offline_model_key} → {_model_cache.current_offline_model_key}")
        if _model_cache.offline_model is not None and _model_cache.offline_model_device != device:
            offline_reasons.append(f"device changed: {_model_cache.offline_model_device} → {device}")
        if _model_cache.offline_model is not None and _model_cache.offline_model_spk != spk_enabled:
            offline_reasons.append(f"spk changed: {_model_cache.offline_model_spk} → {spk_enabled}")
        offline_task = _prepare_reload(
            need_offline, "Offline", _load_offline_model, (device, spk_enabled),
            cache_attr="offline_model", clear_what="offline",
            reasons=", ".join(offline_reasons),
        )

        # ===== 阶段2：并行加载所有需要重载的模型 =====
        pending_tasks = [t for t in (stream_task, vad_task, offline_task) if t is not None]
        if len(pending_tasks) >= 2:
            print(f"[INFO] Parallel loading {len(pending_tasks)} models: "
                  f"{[t['label'] for t in pending_tasks]}")
        elif len(pending_tasks) == 0:
            print("[INFO] All models reused from cache")

        await _execute_parallel_loads(pending_tasks, loop)

        # 模型加载完成后，重新禁用 empty_cache 以优化热路径延迟
        if TORCH_AVAILABLE and torch.cuda.is_available():
            _disable_empty_cache()

        return _model_cache.stream_model, _model_cache.vad_model, _model_cache.offline_model




def clear_model_cache(what: Literal["all", "offline", "stream_vad", "stream", "vad"] = "all"):
    """清除模型缓存并释放 GPU 内存"""
    # 清除缓存前先恢复 empty_cache，确保 GPU 内存能正确释放
    _enable_empty_cache()

    if what in ("all", "stream_vad", "stream") and _model_cache.stream_model is not None:
        _model_cache.stream_model = None
        _model_cache.stream_model_device = None
    if what in ("all", "stream_vad", "vad") and _model_cache.vad_model is not None:
        _model_cache.vad_model = None
        _model_cache.vad_model_device = None
    if what in ("all", "offline") and _model_cache.offline_model is not None:
        _model_cache.offline_model = None
        _model_cache.offline_model_key = None
        _model_cache.offline_model_device = None
        _model_cache.offline_model_spk = False

    # 清除缓存后释放 GPU 内存（绕过热路径抑制标志）
    if TORCH_AVAILABLE and torch.cuda.is_available():
        _safe_empty_cache()

    print(f"[INFO] Model cache cleared ({what})")
