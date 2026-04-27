from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None

from .adaptive_kv_policy import AdaptiveHybridKVCache, AdaptiveKVPolicy
from .blocked_kv_cache import BlockedQuantizedKVCache
from .kv_quant import QuantizedKVCache, floating_kv_memory_bytes
from .utils import synchronize


def _blank_breakdown():
    return {
        "total_decode_ms": 0.0,
        "model_forward_ms": 0.0,
        "importance_scoring_ms": 0.0,
        "quantization_ms": 0.0,
        "dequantization_ms": 0.0,
        "precision_assignment_ms": 0.0,
        "tensor_copy_relayout_ms": 0.0,
    }


@dataclass
class DecodeResult:
    generated_ids: "torch.Tensor"
    step_logits: List["torch.Tensor"]
    seconds: float
    tokens_per_sec: float
    latency_ms_per_token: float
    kv_cache_bytes: int
    text: Optional[str] = None
    compression_ratio: float = 1.0
    precision_counts: Optional[dict] = None
    timing_breakdown: dict = field(default_factory=_blank_breakdown)
    step_timings: List[dict] = field(default_factory=list)


def _restore_cache_format(original_cache, legacy_cache):
    if hasattr(original_cache, "get_seq_length"):
        try:
            from transformers.cache_utils import DynamicCache

            return DynamicCache(legacy_cache)
        except Exception:
            return legacy_cache
    return legacy_cache


def _merge_breakdown(target, update):
    for key, value in (update or {}).items():
        target[key] = target.get(key, 0.0) + float(value)


def _prepare_past_for_mode(
    past_key_values,
    mode: str,
    block_size: int,
    dtype,
    adaptive_policy=None,
    attentions=None,
    step: int = 0,
):
    baseline_bytes = floating_kv_memory_bytes(past_key_values, assume_fp16=True)
    precision_counts = None
    timing = _blank_breakdown()
    if mode == "baseline":
        return past_key_values, baseline_bytes, 1.0, precision_counts, timing
    if mode == "int8":
        t0 = time.perf_counter()
        q_cache = QuantizedKVCache.from_past_key_values(past_key_values, block_size=block_size)
        timing["quantization_ms"] += (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        legacy_cache = q_cache.to_past_key_values(dtype=dtype)
        timing["dequantization_ms"] += (time.perf_counter() - t1) * 1000
        cache_bytes = q_cache.memory_bytes()
        compression = baseline_bytes / cache_bytes if cache_bytes else 0.0
        return _restore_cache_format(past_key_values, legacy_cache), cache_bytes, compression, precision_counts, timing
    if mode == "blocked_int8":
        t0 = time.perf_counter()
        b_cache = BlockedQuantizedKVCache.from_past_key_values(past_key_values, block_size=block_size)
        timing["quantization_ms"] += (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        legacy_cache = b_cache.to_past_key_values(dtype=dtype)
        timing["dequantization_ms"] += (time.perf_counter() - t1) * 1000
        cache_bytes = b_cache.memory_bytes()
        compression = baseline_bytes / cache_bytes if cache_bytes else 0.0
        return _restore_cache_format(past_key_values, legacy_cache), cache_bytes, compression, precision_counts, timing
    if mode == "adaptive":
        policy = adaptive_policy or AdaptiveKVPolicy(block_size=block_size)
        a_cache = AdaptiveHybridKVCache.from_past_key_values(
            past_key_values,
            policy=policy,
            attentions=attentions,
            step=step,
        )
        legacy_cache = a_cache.to_past_key_values(dtype=dtype)
        cache_bytes = a_cache.memory_bytes()
        compression = baseline_bytes / cache_bytes if cache_bytes else 0.0
        precision_counts = a_cache.precision_counts()
        return _restore_cache_format(past_key_values, legacy_cache), cache_bytes, compression, precision_counts, a_cache.timing
    raise ValueError(f"Unknown decode mode: {mode}")


@torch.no_grad() if torch is not None else (lambda fn: fn)
def decode(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    max_new_tokens: int = 32,
    mode: str = "baseline",
    block_size: int = 16,
    temperature: float = 0.0,
    adaptive_policy: AdaptiveKVPolicy | None = None,
) -> DecodeResult:
    """Manual greedy decoding loop with optional KV-cache quantization.

    TODO: For production-quality speedups, custom attention should consume mixed
    precision cache tensors directly. Here we dequantize before each model
    forward because Hugging Face causal LMs expect floating-point caches.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required. Install with `pip install -r requirements.txt`.")

    device = input_ids.device
    model_dtype = next(model.parameters()).dtype
    generated = input_ids
    step_logits = []
    kv_cache_bytes = 0
    compression_ratio = 1.0
    precision_counts = None
    adaptive_policy = adaptive_policy or AdaptiveKVPolicy(block_size=block_size)
    need_attentions = mode == "adaptive" and adaptive_policy.needs_attention
    timing_breakdown = _blank_breakdown()
    step_timings = []

    synchronize(device)
    start = time.perf_counter()

    t_forward = time.perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_attentions=need_attentions)
    synchronize(device)
    timing_breakdown["model_forward_ms"] += (time.perf_counter() - t_forward) * 1000
    past_key_values = outputs.past_key_values
    step_logits.append(outputs.logits[:, -1, :].detach().cpu())

    for step in range(max_new_tokens):
        step_start = time.perf_counter()
        if temperature and temperature > 0:
            probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        past_key_values, kv_cache_bytes, compression_ratio, precision_counts, prepare_timing = _prepare_past_for_mode(
            past_key_values,
            mode,
            block_size,
            model_dtype,
            adaptive_policy=adaptive_policy,
            attentions=getattr(outputs, "attentions", None) if need_attentions else None,
            step=step,
        )
        _merge_breakdown(timing_breakdown, prepare_timing)
        t_forward = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=need_attentions,
        )
        synchronize(device)
        forward_ms = (time.perf_counter() - t_forward) * 1000
        timing_breakdown["model_forward_ms"] += forward_ms
        past_key_values = outputs.past_key_values
        step_logits.append(outputs.logits[:, -1, :].detach().cpu())
        step_latency_ms = (time.perf_counter() - step_start) * 1000
        step_timings.append(
            {
                "step": step,
                "latency_ms": step_latency_ms,
                "tokens_per_sec": 1000.0 / step_latency_ms if step_latency_ms > 0 else 0.0,
                "kv_memory_mb": kv_cache_bytes / (1024**2),
                "model_forward_ms": forward_ms,
                "importance_scoring_ms": prepare_timing.get("importance_scoring_ms", 0.0),
                "quantization_ms": prepare_timing.get("quantization_ms", 0.0),
                "dequantization_ms": prepare_timing.get("dequantization_ms", 0.0),
                "precision_assignment_ms": prepare_timing.get("precision_assignment_ms", 0.0),
                "tensor_copy_relayout_ms": prepare_timing.get("tensor_copy_relayout_ms", 0.0),
            }
        )

    synchronize(device)
    seconds = time.perf_counter() - start
    timing_breakdown["total_decode_ms"] = seconds * 1000
    tokens_per_sec = max_new_tokens / seconds if seconds > 0 else 0.0
    latency = (seconds / max_new_tokens) * 1000 if max_new_tokens else 0.0
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0] if tokenizer is not None else None

    return DecodeResult(
        generated_ids=generated.detach().cpu(),
        step_logits=step_logits,
        seconds=seconds,
        tokens_per_sec=tokens_per_sec,
        latency_ms_per_token=latency,
        kv_cache_bytes=kv_cache_bytes,
        text=text,
        compression_ratio=compression_ratio,
        precision_counts=precision_counts,
        timing_breakdown=timing_breakdown,
        step_timings=step_timings,
    )
