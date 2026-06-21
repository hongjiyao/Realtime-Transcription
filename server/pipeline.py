import asyncio
import time
import traceback

from server.config import SAMPLE_RATE, _app_config
from server.models import _gpu_executor, _cuda_stream_vad
from server import vad


def _consume_and_buffers(session, chunk) -> None:
    """消费一个 chunk 并写入 speech_buffer / vad_buffer"""
    session.audio_buffer.consume(session._chunk_stride)
    session.speech_buffer.write(chunk)
    session.vad_buffer.write(chunk)
    if session.speech_start_ms == 0:
        session.speech_start_ms = session.total_duration_ms


def _needs_vad(session) -> tuple:
    """判断是否需要 VAD 检测。返回 (need_vad, vad_check_samples)"""
    vad_check_interval = _app_config.vad_check_interval_sec
    vad_buffer_sec = len(session.vad_buffer) / SAMPLE_RATE
    elapsed_since_check = session.total_duration_ms - session.vad_last_check_ms
    vad_interval = vad_check_interval * (3 if session.in_speech else 2)
    need_vad = vad_buffer_sec >= vad_check_interval and elapsed_since_check >= vad_interval * 1000
    vad_check_samples = min(len(session.vad_buffer), int(SAMPLE_RATE * 1.5)) if need_vad else 0
    return need_vad, vad_check_samples


async def _send_throttled(session, result) -> None:
    """节流发送 partial 结果，is_final 始终发送"""
    if not result:
        return
    is_final = result.get("is_final", False)
    new_text = result.get("text", "")
    if not (is_final or new_text):
        return
    now = time.time()
    if is_final or (new_text != session._last_sent_partial and now - session._last_partial_send_time >= session._PARTIAL_THROTTLE_SEC):
        session._last_sent_partial = new_text
        session._last_partial_send_time = now
        await session._send_result(result)


def _submit_inference(session, loop, chunk, need_vad: bool, vad_check_samples: int):
    """提交 Pass1 + 可选 VAD 推理任务，返回 (inference_task, vad_task)"""
    inference_task = loop.run_in_executor(_gpu_executor, session.run_streaming_chunk, chunk, False)
    vad_task = None
    if need_vad and vad_check_samples > 0:
        vad_task = loop.run_in_executor(_gpu_executor, vad.run_vad,
                                        session.vad_model, session.vad_buffer.peek_from_end(vad_check_samples),
                                        _cuda_stream_vad)
    return inference_task, vad_task


async def _await_results(session, loop, inference_task, vad_task):
    """等待推理结果，返回 (endpoint_detected, need_break)"""
    need_break = False
    endpoint_detected = False

    if vad_task is not None:
        result = await inference_task
        await _send_throttled(session, result)
        vad_segments = await vad_task
        session.vad_last_check_ms = session.total_duration_ms
        slice_ms = int(min(len(session.vad_buffer), int(SAMPLE_RATE * 1.5)) / SAMPLE_RATE * 1000)
        endpoint_detected, new_in_speech, new_empty, new_last_voiced = vad.process_vad_result(
            vad_segments, slice_ms, session.in_speech, session._vad_empty_count,
            _app_config.vad_silence_gap_sec, session._SPEECH_ACTIVE_THRESHOLD_MS,
            last_voiced_end_ms=session._last_voiced_end_ms,
            current_total_ms=session.total_duration_ms,
        )
        session.in_speech = new_in_speech
        session._vad_empty_count = new_empty
        # 端点检测时不覆盖 _last_voiced_end_ms，保留旧值供 _finalize_utterance 精确计算消费长度
        if not endpoint_detected:
            session._last_voiced_end_ms = new_last_voiced
    else:
        result = await inference_task
        await _send_throttled(session, result)
        endpoint_detected = False

    if endpoint_detected:
        await _finalize_utterance(session, loop, reason="vad")
        need_break = True
    elif len(session.speech_buffer) >= int(session.speech_buffer.capacity * 0.9):
        # 缓冲区溢出保护：当 speech_buffer 接近容量上限（90%）时，强制断句
        # 防止 RingBuffer 静默覆盖最旧音频导致吞音频（连续说话 >9s 无停顿时触发）
        print(f"[WARN] Speech buffer near capacity ({len(session.speech_buffer)}/{session.speech_buffer.capacity}), forcing finalize to prevent audio loss")
        await _finalize_utterance(session, loop, reason="forced")
        need_break = True
    elif session._current_partial_len >= _app_config.max_sentence_chars and session.current_partial:
        await _finalize_utterance(session, loop, reason="forced")
        need_break = True

    return endpoint_detected, need_break


async def try_process_chunks(session) -> None:
    """Pipeline 并行：当前 chunk 推理时，并行准备下一个 chunk"""
    if not session.is_active:
        return
    if len(session.audio_buffer) < session._chunk_stride:
        return

    loop = asyncio.get_running_loop()

    first_chunk = session.audio_buffer.peek_view(session._chunk_stride)
    _consume_and_buffers(session, first_chunk)
    need_vad, vad_check_samples = _needs_vad(session)
    inference_task, vad_task = _submit_inference(session, loop, first_chunk, need_vad, vad_check_samples)

    try:
        while session.is_active:
            has_next = len(session.audio_buffer) >= session._chunk_stride
            next_need_vad, next_vad_samples = False, 0

            if has_next:
                next_chunk = session.audio_buffer.peek_view(session._chunk_stride)
                _consume_and_buffers(session, next_chunk)
                next_need_vad, next_vad_samples = _needs_vad(session)

            _, need_break = await _await_results(session, loop, inference_task, vad_task)
            if need_break:
                break
            if not has_next:
                break

            inference_task, vad_task = _submit_inference(session, loop, next_chunk, next_need_vad, next_vad_samples)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[ERROR] Pipeline crashed: {e}")
        traceback.print_exc()
        session.is_active = False


async def _finalize_utterance(session, loop, reason="vad") -> None:
    """处理语句终结：发 is_final + 触发 Pass2"""
    # 缓存 _last_voiced_end_ms，避免下方 await 期间被并发 VAD 检测修改
    cached_last_voiced_end_ms = session._last_voiced_end_ms

    if reason == "vad":
        print(f"[INFO] VAD endpoint detected, running Pass2 on speech_buffer "
              f"({len(session.speech_buffer)/SAMPLE_RATE:.2f}s)")
    else:
        print(f"[INFO] Sentence too long ({session._current_partial_len} chars), forcing finalize")

    zero_chunk = session._zero_chunk
    result = await loop.run_in_executor(_gpu_executor, session.run_streaming_chunk, zero_chunk, True)
    if result:
        await session._send_result(result)
    session._finalized_text = session.current_partial.strip()
    session._finalized_start_ms = session.speech_start_ms
    session.current_partial = ""
    session._last_sent_partial = ""

    # 统一通过 run_pass2_async 触发：若 Pass2 正在运行，它会发送 fallback 并排队重跑，
    # 而不是取消正在进行的 Pass2（避免丢弃精炼结果）。
    # 注意：必须在 speech_start_ms 重置为 0 之前计算 pass2_buffer_len。
    if (reason == "vad"
            and cached_last_voiced_end_ms > 0
            and session.speech_start_ms > 0):
        # 使用 VAD 的 _last_voiced_end_ms 精确计算消费长度，避免吞掉尾部音频
        speech_duration_ms = cached_last_voiced_end_ms - session.speech_start_ms + 100  # 100ms padding
        pass2_buffer_len = min(len(session.speech_buffer),
                               int(speech_duration_ms * SAMPLE_RATE / 1000))
        if pass2_buffer_len <= 0:
            pass2_buffer_len = len(session.speech_buffer)
    else:
        # reason="forced" 或 _last_voiced_end_ms<=0 或 speech_start_ms<=0：保持原有行为
        pass2_buffer_len = len(session.speech_buffer)
    print(f"[DEBUG] pass2_buffer_len={pass2_buffer_len} "
          f"(speech_buffer total={len(session.speech_buffer)})")

    session.speech_start_ms = 0
    session._last_voiced_end_ms = 0  # 重置，为下一段语音准备
    session.schedule_pass2(pass2_buffer_len=pass2_buffer_len)
