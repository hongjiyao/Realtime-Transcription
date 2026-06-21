import threading
import time
import logging
import numpy as np
from collections import deque

_logger = logging.getLogger(__name__)

try:
    import pyaudiowpatch as pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import torchaudio
except ImportError:
    torchaudio = None
try:
    import torch
except ImportError:
    torch = None


class WasapiAudioCapture:

    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, device_index=None):
        self.target_sample_rate = sample_rate
        self.target_channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index  # None = 自动选择第一个环回设备
        self._stream = None
        self._pyaudio = None
        self._running = threading.Event()
        self._thread = None
        self._source_rate = None
        self._source_channels = None
        self._resampler = None
        self._resample_method = "numpy"
        # 优化：使用 deque 替代 Queue，减少锁开销
        self._audio_queue = deque(maxlen=2000)
        # 预计算的抗混叠滤波器系数（如果需要下采样）
        self._resample_filter = None
        # 预计算的重采样索引（numpy 路径缓存，避免每帧重复创建）
        self._resample_indices = None
        # 事件驱动信号：由捕获线程设置，asyncio 任务等待（外部注入）
        self._data_ready_event = None
        # 预分配 float32 buffer，避免每帧分配临时数组
        self._float32_buffer = None

    def start(self):
        if self._running.is_set():
            _logger.warning("WasapiAudioCapture already running, ignoring start() call")
            return False
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("pyaudiowpatch not available. Install: pip install pyaudiowpatch")

        self._pyaudio = pyaudio.PyAudio()

        try:
            wasapi_info = None
            if self.device_index is not None:
                # 使用指定的设备索引
                try:
                    wasapi_info = self._pyaudio.get_device_info_by_index(self.device_index)
                    _logger.info(f"WASAPI using selected device index={self.device_index}")
                except Exception as e:
                    _logger.warning(f"WASAPI device index {self.device_index} not found: {e}, falling back to auto")

            if wasapi_info is None:
                # 自动选择第一个环回设备
                for loopback in self._pyaudio.get_loopback_device_info_generator():
                    wasapi_info = loopback
                    break

            if wasapi_info is None:
                raise RuntimeError("No WASAPI loopback device found. Make sure you are on Windows with audio playing.")

            self._source_rate = int(wasapi_info["defaultSampleRate"])
            self._source_channels = wasapi_info["maxInputChannels"]
            if self._source_channels <= 0:
                raise RuntimeError(f"设备 {wasapi_info['name']} 的输入通道数为 0，无法捕获音频")
            _logger.info(f"WASAPI device: {wasapi_info.get('name', 'unknown')}, "
                  f"rate={self._source_rate}, channels={self._source_channels}")
        except Exception:
            self._pyaudio.terminate()
            self._pyaudio = None
            raise

        # 初始化重采样器（优先 CPU torchaudio，避免 GPU 往返开销）
        # 2048 样本的重采样在 CPU 上也只需 <1ms，GPU 往返反而更慢
        if self._source_rate != self.target_sample_rate:
            if torchaudio is not None:
                try:
                    self._resampler = torchaudio.transforms.Resample(
                        orig_freq=self._source_rate,
                        new_freq=self.target_sample_rate,
                    )  # 默认在 CPU 上
                    self._resample_method = "torchaudio_cpu"
                    _logger.info(f"Resampler: CPU torchaudio ({self._source_rate}Hz -> {self.target_sample_rate}Hz)")
                except Exception as e:
                    _logger.warning(f"torchaudio Resample not available: {e}, using numpy linear interpolation")
                    self._resampler = None
                    self._resample_method = "numpy"
            else:
                _logger.warning("torchaudio not installed, using numpy linear interpolation")
                self._resampler = None
                self._resample_method = "numpy"

            # 预计算抗混叠滤波器系数（仅当下采样时需要）
            if self._source_rate > self.target_sample_rate:
                cutoff = self.target_sample_rate / (2 * self._source_rate)
                k = max(1, int(8 * self._source_rate / self.target_sample_rate))
                kernel = np.sinc(np.arange(-k, k + 1) * cutoff) * cutoff
                kernel *= np.hamming(len(kernel))
                kernel /= kernel.sum()
                self._resample_filter = kernel.astype(np.float32)
                _logger.info(f"Resample filter precomputed, kernel size={len(kernel)}")

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self._source_channels,
            rate=self._source_rate,
            input=True,
            input_device_index=wasapi_info["index"],
            frames_per_buffer=self.chunk_size,
            stream_callback=None,
        )

        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        _logger.info("WASAPI capture thread started")
        return True

    def _capture_loop(self):
        chunk_count = 0
        _float_scale = np.float32(1.0 / 32768.0)
        # CUDA 同步屏障：确保 PortAudio 的 native stream.read() 调用不会与
        # PyTorch CUDA kernel 在不同线程上并发执行（Windows WASAPI loopback
        # 的 PortAudio C 层与 CUDA 执行引擎存在调度竞争，会触发 0xC0000005）。
        # 优化：用 torch.cuda.Event 非阻塞查询 GPU 空闲状态（~0.001ms），
        # 仅在 GPU 繁忙且距上次同步 >200ms 时才执行全局 synchronize()。
        # GPU 空闲时零同步开销，GPU 繁忙时最多每 200ms 同步一次。
        _cuda_sync = None
        _gpu_events = None
        if torch is not None and torch.cuda.is_available():
            _cuda_sync = torch.cuda.synchronize
            try:
                from server.models import _gpu_event_pass1, _gpu_event_vad, _gpu_event_pass2
                if _gpu_event_pass1 is not None:
                    _gpu_events = (_gpu_event_pass1, _gpu_event_vad, _gpu_event_pass2)
            except ImportError:
                pass
        _last_sync_time = time.time()
        _SYNC_INTERVAL = 0.2  # 200ms
        while self._running.is_set():
            try:
                # GPU 空闲检测：非阻塞查询所有 Event
                gpu_idle = True
                if _gpu_events is not None:
                    try:
                        for evt in _gpu_events:
                            if not evt.query():
                                gpu_idle = False
                                break
                    except Exception:
                        gpu_idle = False  # 查询失败，保守同步
                else:
                    gpu_idle = False  # Event 未初始化，保守同步

                if not gpu_idle:
                    now = time.time()
                    if now - _last_sync_time >= _SYNC_INTERVAL:
                        if _cuda_sync is not None:
                            try:
                                _cuda_sync()
                            except Exception:
                                pass
                        _last_sync_time = now
                else:
                    _last_sync_time = time.time()  # GPU 空闲，重置计时器

                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                # 优化：预分配 buffer + 原地乘法，避免每帧临时数组分配
                pcm16_view = np.frombuffer(data, dtype=np.int16)
                n = len(pcm16_view)
                if self._float32_buffer is None or len(self._float32_buffer) < n:
                    self._float32_buffer = np.empty(n, dtype=np.float32)
                audio = self._float32_buffer[:n]
                np.multiply(pcm16_view, _float_scale, out=audio)

                if self._source_rate != self.target_sample_rate:
                    # 通道混合
                    if self._source_channels > 1:
                        audio = audio.reshape(-1, self._source_channels).mean(axis=1)
                    # 重采样（CPU torchaudio 或 numpy 回退）
                    audio = self._resample(
                        audio, self._source_rate, self.target_sample_rate,
                        filter_kernel=self._resample_filter,
                    )
                else:
                    # 不需要重采样，只做通道混合
                    if self._source_channels > 1:
                        audio = audio.reshape(-1, self._source_channels)
                        audio = audio.mean(axis=1)

                # 优化：使用 deque.append 替代 Queue.put_nowait，无锁操作
                # deque(maxlen=2000) 会自动丢弃最旧的元素，无需人工检测溢出
                self._audio_queue.append(audio)
                if self._data_ready_event is not None:
                    self._data_ready_event.set()  # 通知 asyncio 任务

                chunk_count += 1
                if chunk_count % 200 == 1:
                    qsize = len(self._audio_queue)
                    _logger.debug(f"WASAPI captured {chunk_count} chunks, queue size={qsize}")
            except (OSError, IOError) as e:
                # 流已关闭（Errno -9988）/ 已停止 / 设备断开，均不可恢复
                msg = str(e)
                if ("Stream closed" in msg or "Stream is not open" in msg
                        or "Stream is stopped" in msg
                        or getattr(e, "errno", 0) == -9988):
                    _logger.info(f"Audio stream closed/stopped, stopping capture: {e}")
                    break
                # stop() 主动停止后出现的任何 I/O 错误都应直接退出，避免再触碰 stream
                if not self._running.is_set():
                    _logger.info(f"Capture stopping after I/O error: {e}")
                    break
                # 其他可恢复的 I/O 错误
                _logger.warning(f"Audio capture I/O error: {e}")
                time.sleep(0.1)
                continue
            except Exception as e:
                # 不可恢复错误
                _logger.error(f"Audio capture fatal error: {e}")
                break
        _logger.info(f"WASAPI capture loop ended, total chunks={chunk_count}")

    def _resample(self, audio: np.ndarray, orig_rate: int, target_rate: int,
                  filter_kernel=None):
        """重采样音频数据（CPU torchaudio 优先，numpy 线性插值回退）

        首次 torchaudio 失败时永久切换为 numpy，后续不再重试。

        Args:
            filter_kernel: 预计算的抗混叠滤波器系数，None 则不施加抗混叠滤波

        Returns:
            重采样后的 np.ndarray
        """
        if orig_rate == target_rate:
            return audio

        if self._resample_method == "torchaudio_cpu" and self._resampler is not None:
            try:
                audio_tensor = torch.from_numpy(audio)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                resampled = self._resampler(audio_tensor).squeeze(0).detach()
                return resampled.numpy()
            except Exception as e:
                _logger.warning(f"CPU torchaudio resample failed: {e}, permanently switching to numpy")
                self._resample_method = "numpy"
                self._resampler = None

        # numpy 线性插值（使用缓存的索引避免每帧重复创建）
        new_len = int(len(audio) * target_rate / orig_rate)
        if new_len <= 0:
            return np.empty(0, dtype=np.float32)
        if self._resample_indices is None or len(self._resample_indices) != new_len:
            self._resample_indices = np.linspace(0, len(audio) - 1, new_len)
        resampled = np.interp(self._resample_indices, np.arange(len(audio)), audio)
        if target_rate < orig_rate and filter_kernel is not None:
            resampled = np.convolve(resampled, filter_kernel, mode='same')
        return resampled.astype(np.float32)

    def read(self):
        # 优化：使用 deque.popleft 替代 Queue.get_nowait，无锁操作
        if self._audio_queue:
            return self._audio_queue.popleft()
        return None

    def is_capturing(self):
        return self._running.is_set() and self._thread is not None and self._thread.is_alive()

    def stop(self):
        # 正确的关闭顺序（修复 WASAPI loopback 下 capture 线程在 stream.read() 内
        # access violation 的根因）：
        #
        # 旧实现先 _thread.join(timeout=2) 再 stop_stream()/close()。但 PortAudio 的
        # WASAPI loopback 阻塞式 read() 不会因 _running.clear() 而提前返回——它仍阻塞
        # 在 native Pa_ReadStream 中等待下一个音频块（~10ms 周期，loopback 无声音时
        # 可能远超 join 的 2s timeout）。join 超时后主线程继续执行 stop_stream()/close()，
        # 而此时 capture 线程仍阻塞在 read() 内部，PortAudio native 层在被 close 的
        # 同时另一个线程还在读 → 0xC0000005 access violation。
        #
        # 修复：先 stop_stream() 让 PortAudio 主动停止捕获并解除 capture 线程的阻塞
        # read（read 会抛 "Stream is stopped" 异常或返回，capture loop 的 except 会
        # 正常退出循环），确保 capture 线程在 close() 之前已完全离开 native read，
        # 再 join，最后 close/terminate。
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception as e:
                _logger.warning(f"Error stopping stream: {e}")
        # 现在 capture 线程的阻塞 read 已被解除，可安全等待它退出
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.warning("WASAPI capture thread did not exit within 5s")
            self._thread = None
        # 确认 capture 线程已离开 read() 后，再关闭流和 PyAudio
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as e:
                _logger.warning(f"Error closing stream: {e}")
            self._stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception as e:
                _logger.warning(f"Error during cleanup: {e}")
            self._pyaudio = None
        self._audio_queue.clear()
        # 清理重采样器
        self._resampler = None
        _logger.info("WASAPI capture stopped and cleaned up")

    @staticmethod
    def is_available():
        return PYAUDIO_AVAILABLE

    @staticmethod
    def list_devices():
        """列出所有 WASAPI 环回设备（即系统音频输出设备）"""
        if not PYAUDIO_AVAILABLE:
            return []
        pa = None
        try:
            pa = pyaudio.PyAudio()
            devices = []
            for loopback in pa.get_loopback_device_info_generator():
                devices.append({
                    "index": loopback["index"],
                    "name": loopback.get("name", "unknown"),
                    "sample_rate": int(loopback.get("defaultSampleRate", 48000)),
                    "channels": loopback.get("maxInputChannels", 2),
                })
            return devices
        except Exception as e:
            _logger.error(f"Failed to list audio devices: {e}")
            return []
        finally:
            if pa is not None:
                pa.terminate()
