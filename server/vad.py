import traceback

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from server.config import _normalize_timestamp_ms
from server.models import _gpu_event_vad


def run_vad(vad_model, audio_data: np.ndarray, cuda_stream_vad) -> list:
    """运行 VAD 模型推理，返回语音段列表 [[start_ms, end_ms], ...]"""
    try:
        with torch.no_grad():
            if cuda_stream_vad is not None:
                with torch.cuda.stream(cuda_stream_vad):
                    res = vad_model.generate(input=audio_data, batch_size=1)
                _gpu_event_vad.record()
            else:
                res = vad_model.generate(input=audio_data, batch_size=1)

        segments = []
        if res and len(res) > 0:
            vad_result = res[0].get("value", [])
            for seg in vad_result:
                if isinstance(seg, list) and len(seg) >= 2:
                    segments.append([seg[0], seg[1]])
                elif isinstance(seg, dict):
                    start = _normalize_timestamp_ms(seg.get("start", seg.get("start_ms", 0)))
                    end = _normalize_timestamp_ms(seg.get("end", seg.get("end_ms", 0)))
                    segments.append([int(start), int(end)])

        return segments
    except (RuntimeError, torch.cuda.OutOfMemoryError, torch.cuda.CudaError) as e:
        print(f"[ERROR] VAD fatal error: {e}")
        traceback.print_exc()
        raise
    except (TypeError, KeyError, ValueError) as e:
        print(f"[WARN] VAD result parsing issue (returning empty): {e}")
        return []
    except Exception as e:
        print(f"[ERROR] VAD unexpected error: {e}")
        traceback.print_exc()
        return []


def process_vad_result(segments: list, slice_ms: int, in_speech: bool,
                       vad_empty_count: int, vad_silence_gap: float,
                       speech_active_threshold_ms: int,
                       last_voiced_end_ms: float = 0.0,
                       current_total_ms: int = 0) -> tuple:
    """处理 VAD 语音段结果，检测端点。

    基于【绝对时间轴】累积静音，避免滚动窗口大小限制导致静音间隔被掩盖。
    返回 (endpoint_detected, new_in_speech, new_vad_empty_count, new_last_voiced_end_ms)。

    Args:
        segments: VAD 返回的语音段列表 [[start_ms, end_ms], ...]（切片内相对 ms）
        slice_ms: 输入切片的时长（毫秒）
        in_speech: 当前是否在语音中
        vad_empty_count: 当前连续空 VAD 段计数
        vad_silence_gap: 静音间隔阈值（秒）
        speech_active_threshold_ms: 语音激活阈值（毫秒）
        last_voiced_end_ms: 上一次检测到语音的【绝对】结束时刻（ms）；0 表示尚无语音
        current_total_ms: 当前音频流的绝对总时长（ms），用作切片右边界对齐
    """
    silence_threshold_ms = vad_silence_gap * 1000
    # 切片右边界对齐到绝对时间轴：切片末尾 == current_total_ms
    slice_end_abs = current_total_ms

    # 切片内有语音段：更新"最后语音绝对结束时刻"为切片内最后一段的绝对结束
    if segments:
        last_seg = segments[-1]
        rel_end_ms = last_seg[1]
        # 切片内相对时间 → 绝对：切片右边界为 slice_end_abs，左边界为 slice_end_abs - slice_ms
        new_last_voiced_end = slice_end_abs - (slice_ms - rel_end_ms)
        if last_voiced_end_ms <= 0:
            new_last_voiced_end_ms = new_last_voiced_end
        else:
            new_last_voiced_end_ms = max(last_voiced_end_ms, new_last_voiced_end)

        # 当前是否在语音：切片尾部仍有语音（尾部静音 < 激活阈值）
        silence_tail = slice_ms - rel_end_ms
        currently_in_speech = silence_tail < speech_active_threshold_ms
        return False, currently_in_speech, 0, new_last_voiced_end_ms

    # 切片内无语音段：累积静音
    if last_voiced_end_ms > 0:
        silence_ms = slice_end_abs - last_voiced_end_ms
        if silence_ms >= silence_threshold_ms:
            return True, False, 0, 0.0
        # 尚未达到阈值：保留 last_voiced_end_ms 继续累积
        return False, False, vad_empty_count + 1, last_voiced_end_ms

    # 从未检测到语音且本切片也空：不触发端点
    return False, False, 0, 0.0
