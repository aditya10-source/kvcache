from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import ceil
from typing import List, Sequence, Tuple

try:
    import torch
except ImportError:
    torch = None

from .importance_score import compute_importance_scores
from .kv_quant import PastKeyValues, iter_kv_layers


PRECISION_FP16 = "fp16"
PRECISION_INT8 = "int8"
PRECISION_INT4 = "int4"
_PRECISION_RANK = {PRECISION_INT4: 0, PRECISION_INT8: 1, PRECISION_FP16: 2}
_RANK_PRECISION = {value: key for key, value in _PRECISION_RANK.items()}


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required. Install with `pip install -r requirements.txt`.")


def _empty_timing():
    return {
        "importance_scoring_ms": 0.0,
        "precision_assignment_ms": 0.0,
        "quantization_ms": 0.0,
        "dequantization_ms": 0.0,
        "tensor_copy_relayout_ms": 0.0,
    }


@dataclass
class AdaptiveKVPolicy:
    block_size: int = 16
    fp16_ratio: float = 0.2
    int8_ratio: float = 0.5
    importance_mode: str = "recency"
    alpha: float = 0.5
    beta: float = 0.5
    update_interval: int = 16
    monotonic: bool = True
    cached_precisions: List[str] = field(default_factory=list)
    last_update_step: int = -1

    @property
    def needs_attention(self) -> bool:
        return self.importance_mode in {"attention", "hybrid"}

    def _should_recompute(self, step: int, num_blocks: int) -> bool:
        if not self.cached_precisions:
            return True
        if len(self.cached_precisions) < num_blocks:
            return True
        if self.update_interval <= 1:
            return True
        return (step - self.last_update_step) >= self.update_interval

    def assign_precisions(self, seq_len: int, attentions=None, device=None, step: int = 0):
        _require_torch()
        timing = _empty_timing()
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        if num_blocks <= 0:
            return [], timing, False

        if not self._should_recompute(step, num_blocks):
            return self.cached_precisions[:num_blocks], timing, False

        t0 = time.perf_counter()
        scores = compute_importance_scores(
            seq_len=seq_len,
            block_size=self.block_size,
            attentions=attentions,
            alpha=self.alpha,
            beta=self.beta,
            device=device,
        )
        if self.importance_mode == "recency":
            values = scores.recency
        elif self.importance_mode == "attention":
            values = scores.attention
        elif self.importance_mode == "hybrid":
            values = scores.hybrid
        else:
            raise ValueError(f"Unknown importance_mode: {self.importance_mode}")
        timing["importance_scoring_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        fp16_count = min(num_blocks, max(0, ceil(num_blocks * self.fp16_ratio)))
        int8_count = min(num_blocks - fp16_count, max(0, ceil(num_blocks * self.int8_ratio)))
        ranks = torch.full((num_blocks,), _PRECISION_RANK[PRECISION_INT4], dtype=torch.int64, device=values.device)
        order = torch.argsort(values, descending=True)
        if fp16_count:
            ranks[order[:fp16_count]] = _PRECISION_RANK[PRECISION_FP16]
        if int8_count:
            ranks[order[fp16_count : fp16_count + int8_count]] = _PRECISION_RANK[PRECISION_INT8]

        if self.monotonic and self.cached_precisions:
            old = torch.tensor(
                [_PRECISION_RANK[p] for p in self.cached_precisions[:num_blocks]],
                dtype=torch.int64,
                device=values.device,
            )
            if old.numel() < num_blocks:
                pad = torch.full((num_blocks - old.numel(),), _PRECISION_RANK[PRECISION_FP16], dtype=torch.int64, device=values.device)
                old = torch.cat([old, pad])
            ranks = torch.minimum(ranks, old)

        rank_list = ranks.detach().cpu().tolist()
        precisions = [_RANK_PRECISION[int(rank)] for rank in rank_list]
        self.cached_precisions = precisions
        self.last_update_step = step
        timing["precision_assignment_ms"] = (time.perf_counter() - t1) * 1000
        return precisions, timing, True


@dataclass
class AdaptiveBlock:
    precision: str
    values: object
    scale: object | None
    valid_tokens: int

    def memory_bytes(self) -> int:
        if self.precision == PRECISION_FP16:
            return self.values.numel() * 2
        if self.precision == PRECISION_INT8:
            return self.values.numel() + self.scale.numel() * self.scale.element_size()
        if self.precision == PRECISION_INT4:
            return ceil(self.values.numel() / 2) + self.scale.numel() * self.scale.element_size()
        raise ValueError(f"Unknown precision: {self.precision}")


@dataclass
class AdaptiveTensor:
    blocks: List[AdaptiveBlock]
    original_shape: Tuple[int, ...]
    block_size: int
    dtype_name: str

    def memory_bytes(self) -> int:
        return sum(block.memory_bytes() for block in self.blocks)


def _quantize_block(block, precision: str) -> AdaptiveBlock:
    if precision == PRECISION_FP16:
        return AdaptiveBlock(precision=precision, values=block.to(torch.float16).contiguous(), scale=None, valid_tokens=block.shape[2])

    quant_max = 127 if precision == PRECISION_INT8 else 7
    max_abs = block.abs().amax(dim=(2, 3), keepdim=True)
    scale = (max_abs / quant_max).clamp(min=1e-8)
    values = torch.round(block / scale).clamp(-quant_max, quant_max).to(torch.int8).contiguous()
    return AdaptiveBlock(precision=precision, values=values, scale=scale.squeeze(-1).contiguous(), valid_tokens=block.shape[2])


def adaptive_quantize_tensor(x, precisions: Sequence[str], block_size: int, timing=None) -> AdaptiveTensor:
    _require_torch()
    if x.ndim != 4:
        raise ValueError(f"Expected [B, H, S, D], got shape {tuple(x.shape)}")
    bsz, heads, seq_len, head_dim = x.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    if len(precisions) != num_blocks:
        raise ValueError(f"Expected {num_blocks} precision assignments, got {len(precisions)}")

    t0 = time.perf_counter()
    blocks = []
    for block_idx, precision in enumerate(precisions):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        block = x[:, :, start:end, :]
        blocks.append(_quantize_block(block, precision))
    if timing is not None:
        timing["quantization_ms"] += (time.perf_counter() - t0) * 1000

    return AdaptiveTensor(
        blocks=blocks,
        original_shape=(bsz, heads, seq_len, head_dim),
        block_size=block_size,
        dtype_name=str(x.dtype).replace("torch.", ""),
    )


def dequantize_adaptive_tensor(tensor: AdaptiveTensor, dtype=None, timing=None):
    _require_torch()
    t0 = time.perf_counter()
    pieces = []
    for block in tensor.blocks:
        if block.precision == PRECISION_FP16:
            out = block.values
        else:
            scale = block.scale.unsqueeze(-1).to(torch.float32)
            out = block.values.to(torch.float32) * scale
        pieces.append(out[:, :, : block.valid_tokens, :])
    result = torch.cat(pieces, dim=2).contiguous() if pieces else torch.empty(tensor.original_shape)
    if timing is not None:
        timing["dequantization_ms"] += (time.perf_counter() - t0) * 1000
    if dtype is None:
        dtype = getattr(torch, tensor.dtype_name, torch.float32)
    return result.to(dtype)


@dataclass
class AdaptiveHybridKVCache:
    layers: List[Tuple[AdaptiveTensor, AdaptiveTensor]]
    precisions: List[str]
    block_size: int = 16
    timing: dict = field(default_factory=_empty_timing)
    recomputed: bool = False

    @classmethod
    def from_past_key_values(
        cls,
        past_key_values: PastKeyValues,
        policy: AdaptiveKVPolicy,
        attentions=None,
        step: int = 0,
    ) -> "AdaptiveHybridKVCache":
        timing = _empty_timing()
        t_relayout = time.perf_counter()
        kv_layers = list(iter_kv_layers(past_key_values))
        timing["tensor_copy_relayout_ms"] += (time.perf_counter() - t_relayout) * 1000
        if not kv_layers:
            return cls(layers=[], precisions=[], block_size=policy.block_size, timing=timing)
        seq_len = kv_layers[0][0].shape[2]
        precisions, assign_timing, recomputed = policy.assign_precisions(
            seq_len=seq_len,
            attentions=attentions,
            device=kv_layers[0][0].device,
            step=step,
        )
        for key, value in assign_timing.items():
            timing[key] += value
        layers = []
        for key, value in kv_layers:
            layers.append((
                adaptive_quantize_tensor(key, precisions, policy.block_size, timing=timing),
                adaptive_quantize_tensor(value, precisions, policy.block_size, timing=timing),
            ))
        return cls(layers=layers, precisions=precisions, block_size=policy.block_size, timing=timing, recomputed=recomputed)

    def to_past_key_values(self, dtype=None):
        return tuple(
            (
                dequantize_adaptive_tensor(key, dtype, timing=self.timing),
                dequantize_adaptive_tensor(value, dtype, timing=self.timing),
            )
            for key, value in self.layers
        )

    def memory_bytes(self) -> int:
        return sum(key.memory_bytes() + value.memory_bytes() for key, value in self.layers)

    def precision_counts(self):
        return {
            PRECISION_FP16: self.precisions.count(PRECISION_FP16),
            PRECISION_INT8: self.precisions.count(PRECISION_INT8),
            PRECISION_INT4: self.precisions.count(PRECISION_INT4),
        }
