import threading
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from server.config import (
    TORCH_AVAILABLE, _normalize_timestamp_ms, _app_config,
)


class SpeakerBank:
    """跨语句说话人标签一致性映射

    问题：每次 Pass2 调用中，FunASR 的聚类标签从 0 开始重新编号。
    例如：第一次 Pass2 说话人 A=0, B=1；第二次 Pass2 说话人 B=0, A=1。
    这导致同一说话人在不同句子中获得不同编号。

    解决方案：
    - 维护已知说话人的中心嵌入向量（来自 campplus 模型的 192 维 embedding）
    - 每次 Pass2 完成后，将模型原生聚类标签映射到全局一致标签
    - 映射方式：计算每个原生说话人的平均嵌入，与已知中心比较
    - 余弦相似度超过阈值 → 映射到已有说话人
    - 低于阈值 → 创建新说话人

    注意：不修改模型内部的聚类结果，只做标签映射。

    GPU 优化：所有嵌入计算在 GPU 上完成，避免 GPU→CPU→numpy 转换开销。
    """

    def __init__(self, max_speakers=20):
        self.max_speakers = max_speakers
        self.centers = []
        self.counts = []
        self._device = None
        self._centers_tensor_cache = None
        self._lock = threading.Lock()
        self._SEG_SHIFT_MS = 750
        self._SEG_DUR_MS = 1500

    def _ensure_device(self):
        if self._device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self._device = torch.device('cuda:0')
            else:
                self._device = torch.device('cpu')

    def _invalidate_centers_cache(self):
        self._centers_tensor_cache = None

    def _get_centers_tensor(self):
        if self._centers_tensor_cache is None and len(self.centers) > 0:
            self._centers_tensor_cache = torch.stack(self.centers)
        return self._centers_tensor_cache

    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        norm = v.norm()
        if norm < 1e-8:
            return v
        return v / norm

    def _to_tensor(self, data) -> torch.Tensor:
        self._ensure_device()
        if isinstance(data, torch.Tensor):
            return data.to(self._device, non_blocking=True)
        return torch.from_numpy(np.asarray(data, dtype=np.float32)).to(self._device, non_blocking=True)

    def _collect_local_spk_indices(self, sentence_info: list) -> dict:
        local_spk_to_indices = {}
        for idx, s in enumerate(sentence_info):
            if isinstance(s, dict):
                local_spk = s.get("spk", 0)
            else:
                continue
            if local_spk not in local_spk_to_indices:
                local_spk_to_indices[local_spk] = []
            local_spk_to_indices[local_spk].append(idx)
        return local_spk_to_indices

    def _compute_local_spk_embeddings(self, sentence_info: list, spk_embedding, local_spk_to_indices: dict) -> dict:
        local_spk_embeddings = {}
        if not local_spk_to_indices:
            return local_spk_embeddings

        spk_emb = self._to_tensor(spk_embedding)
        if len(spk_emb) == 0:
            return local_spk_embeddings

        n_emb = len(spk_emb)
        seg_shift_ms = self._SEG_SHIFT_MS
        seg_dur_ms = self._SEG_DUR_MS

        sentence_emb_indices = {}
        for i, s in enumerate(sentence_info):
            if not isinstance(s, dict):
                continue
            s_start = _normalize_timestamp_ms(s.get("start", s.get("start_ms", 0)))
            s_end = _normalize_timestamp_ms(s.get("end", s.get("end_ms", 0)))
            s_start = int(s_start)
            s_end = int(s_end)

            j_indices = np.arange(n_emb)
            emb_starts = j_indices * seg_shift_ms
            emb_ends = emb_starts + seg_dur_ms
            overlaps = np.maximum(0, np.minimum(s_end, emb_ends) - np.maximum(s_start, emb_starts))
            indices = j_indices[overlaps > 0].tolist()
            sentence_emb_indices[i] = indices

        for local_spk, sent_indices in local_spk_to_indices.items():
            all_emb_indices = []
            for si in sent_indices:
                all_emb_indices.extend(sentence_emb_indices.get(si, []))
            if all_emb_indices:
                unique_indices = list(set(all_emb_indices))
                idx_tensor = torch.tensor(unique_indices, device=self._device)
                avg_emb = spk_emb[idx_tensor].mean(dim=0)
                local_spk_embeddings[local_spk] = avg_emb
            else:
                local_spk_embeddings[local_spk] = spk_emb.mean(dim=0)

        return local_spk_embeddings

    def _map_local_to_global(self, local_spk_embeddings: dict, preset_spk_num: int, sim_threshold: float) -> dict:
        """P-09: 将 GPU 计算移到锁外，锁内仅做列表更新"""
        local_to_global = {}
        ema_alpha = _app_config.spk_ema_alpha

        # 锁外：预计算所有相似度
        similarity_results = {}
        with self._lock:
            centers_snapshot = list(self.centers)
            counts_snapshot = list(self.counts)

        if centers_snapshot:
            center_stack = torch.stack(centers_snapshot)
            for local_spk, emb in local_spk_embeddings.items():
                sims = torch.nn.functional.cosine_similarity(
                    center_stack, emb.unsqueeze(0).expand_as(center_stack), dim=1
                )
                best_idx = int(sims.argmax().item())
                best_sim = float(sims[best_idx].item())
                similarity_results[local_spk] = (best_idx, best_sim)

        # 锁内：仅做列表更新
        with self._lock:
            for local_spk, emb in local_spk_embeddings.items():
                if len(self.centers) == 0:
                    self.centers.append(self._normalize(emb.clone()))
                    self._invalidate_centers_cache()
                    self.counts.append(1)
                    local_to_global[local_spk] = 0
                    continue

                if local_spk in similarity_results:
                    best_idx, best_sim = similarity_results[local_spk]
                else:
                    # Fallback: 如果 centers 在锁外快照后发生了变化
                    center_stack = self._get_centers_tensor()
                    sims = torch.nn.functional.cosine_similarity(
                        center_stack, emb.unsqueeze(0).expand_as(center_stack), dim=1
                    )
                    best_idx = int(sims.argmax().item())
                    best_sim = float(sims[best_idx].item())

                force_new = (len(self.centers) < preset_spk_num
                             and best_sim < sim_threshold * 0.8)

                if best_sim >= sim_threshold and not force_new:
                    self.centers[best_idx] = self._normalize(
                        (1 - ema_alpha) * self.centers[best_idx] + ema_alpha * self._normalize(emb)
                    )
                    self._invalidate_centers_cache()
                    self.counts[best_idx] += 1
                    local_to_global[local_spk] = best_idx
                else:
                    speaker_limit = preset_spk_num
                    if len(self.centers) >= speaker_limit:
                        self.centers[best_idx] = self._normalize(
                            (1 - ema_alpha) * self.centers[best_idx] + ema_alpha * self._normalize(emb)
                        )
                        self._invalidate_centers_cache()
                        self.counts[best_idx] += 1
                        local_to_global[local_spk] = best_idx
                    else:
                        new_idx = len(self.centers)
                        self.centers.append(self._normalize(emb.clone()))
                        self._invalidate_centers_cache()
                        self.counts.append(1)
                        local_to_global[local_spk] = new_idx
                        print(f"[INFO] SpeakerBank: new speaker SPK{new_idx} "
                              f"(best_sim={best_sim:.3f}, threshold={sim_threshold})")

        return local_to_global

    def _apply_spk_mapping(self, sentence_info: list, local_to_global: dict) -> list:
        for s in sentence_info:
            if isinstance(s, dict) and "spk" in s:
                old_spk = s["spk"]
                s["spk"] = local_to_global.get(old_spk, old_spk)
        return sentence_info

    def relabel_sentences(self, sentence_info: list, spk_embedding) -> list:
        if spk_embedding is None:
            return sentence_info

        spk_emb = self._to_tensor(spk_embedding)
        if len(spk_emb) == 0:
            return sentence_info

        preset_spk_num = _app_config.preset_spk_num
        sim_threshold = _app_config.spk_sim_threshold

        local_spk_to_indices = self._collect_local_spk_indices(sentence_info)

        if not local_spk_to_indices:
            return sentence_info

        local_spk_embeddings = self._compute_local_spk_embeddings(sentence_info, spk_embedding, local_spk_to_indices)
        local_to_global = self._map_local_to_global(local_spk_embeddings, preset_spk_num, sim_threshold)
        return self._apply_spk_mapping(sentence_info, local_to_global)

    def reset(self):
        with self._lock:
            self.centers = []
            self.counts = []
            self._invalidate_centers_cache()

    def get_info(self) -> dict:
        with self._lock:
            return {
                "num_speakers": len(self.centers),
                "counts": list(self.counts),
            }

    def save_to_dict(self) -> dict:
        """P21: 批量 GPU→CPU 转换，单次 cudaMemcpy 替代 N 次逐元素传输"""
        with self._lock:
            self._ensure_device()
            if self.centers:
                # torch.stack + 批量 cpu() — 单次 D2H 传输
                stacked = torch.stack(self.centers)
                centers_list = stacked.detach().cpu().numpy().tolist()
            else:
                centers_list = []
            return {
                "centers": centers_list,
                "counts": list(self.counts),
                "max_speakers": self.max_speakers,
            }

    def load_from_dict(self, data: dict, config=None) -> int:
        self._ensure_device()

        centers_data = data.get("centers", [])
        counts_data = data.get("counts", [])
        max_spk = data.get("max_speakers", self.max_speakers)

        if not isinstance(centers_data, list):
            raise ValueError(f"centers 数据类型错误: 期望 list，实际 {type(centers_data).__name__}")
        if not isinstance(counts_data, list):
            raise ValueError(f"counts 数据类型错误: 期望 list，实际 {type(counts_data).__name__}")

        max_spk = max(1, min(20, int(max_spk)))

        new_centers = []
        for i, c_list in enumerate(centers_data):
            if i >= max_spk:
                break
            if not isinstance(c_list, list):
                print(f"[WARN] Skipping center[{i}]: not a list (got {type(c_list).__name__})")
                continue
            if len(c_list) != 192:
                print(f"[WARN] Skipping center[{i}]: invalid dimension {len(c_list)} (expected 192)")
                continue
            tensor = torch.tensor(c_list, dtype=torch.float32, device=self._device)
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=0)
            new_centers.append(tensor)

        if len(new_centers) == 0:
            raise ValueError("说话人库为空（没有有效的说话人数据）")

        counts_data = list(counts_data)
        if len(counts_data) < len(new_centers):
            counts_data.extend([1] * (len(new_centers) - len(counts_data)))
        new_counts = counts_data[:len(new_centers)]

        with self._lock:
            self.max_speakers = max_spk
            self.centers = new_centers
            self.counts = new_counts
            self._centers_tensor_cache = None
            if config is not None:
                config.preset_spk_num = max(1, min(20, len(new_centers)))
            return len(self.centers)
