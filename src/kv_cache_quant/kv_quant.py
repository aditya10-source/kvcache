from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import torch
except ImportError:  # Allows docs/tools to import the module before installing deps.
    torch = None


PastKeyValues = Sequence[Tuple["torch.Tensor", "torch.Tensor"]]


@dataclass
class QuantizedTensor:
    values: "torch.Tensor"
    scales: "torch.Tensor"
    original_shape: Tuple[int, ...]
    block_size: int
    dtype_name: str

    def memory_bytes(self) -> int:
        return self.values.numel() * self.values.element_size() + self.scales.numel() * self.scales.element_size()


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required. Install with `pip install -r requirements.txt`.")


def iter_kv_layers(past_key_values):
    """Yield `(key, value)` tensors from old and new Hugging Face cache formats."""

    _require_torch()
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        past_key_values = zip(past_key_values.key_cache, past_key_values.value_cache)

    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            tensors = [item for item in layer if torch.is_tensor(item)]
            if len(tensors) >= 2:
                yield tensors[0], tensors[1]
                continue
        key = getattr(layer, "key", None)
        value = getattr(layer, "value", None)
        if torch.is_tensor(key) and torch.is_tensor(value):
            yield key, value
            continue
        raise ValueError(f"Unsupported past_key_values layer format: {type(layer)!r}")


def quantize_tensor_per_block(x: "torch.Tensor", block_size: int = 16) -> QuantizedTensor:
    """Symmetric INT8 quantization over sequence blocks.

    Expected KV shape is `[B, H, S, D]`. Scales are stored per `[B, H, block, 1]`.
    """

    _require_torch()
    if x.ndim != 4:
        raise ValueError(f"Expected [B, H, S, D], got shape {tuple(x.shape)}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    bsz, heads, seq_len, head_dim = x.shape
    pad = (block_size - seq_len % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    num_blocks = x.shape[2] // block_size
    blocked = x.reshape(bsz, heads, num_blocks, block_size, head_dim)

    max_abs = blocked.abs().amax(dim=(3, 4), keepdim=True)
    scales = (max_abs / 127.0).clamp(min=1e-8)
    q = torch.round(blocked / scales).clamp(-127, 127).to(torch.int8)
    q = q.reshape(bsz, heads, num_blocks * block_size, head_dim)
    if pad:
        q = q[:, :, :seq_len, :]

    return QuantizedTensor(
        values=q.contiguous(),
        scales=scales.squeeze(-1).contiguous(),
        original_shape=(bsz, heads, seq_len, head_dim),
        block_size=block_size,
        dtype_name=str(x.dtype).replace("torch.", ""),
    )


def dequantize_tensor(qt: QuantizedTensor, dtype=None) -> "torch.Tensor":
    _require_torch()
    bsz, heads, seq_len, head_dim = qt.original_shape
    pad = (qt.block_size - seq_len % qt.block_size) % qt.block_size
    values = qt.values
    if pad:
        values = torch.nn.functional.pad(values, (0, 0, 0, pad))
    num_blocks = values.shape[2] // qt.block_size
    blocked = values.reshape(bsz, heads, num_blocks, qt.block_size, head_dim).to(torch.float32)
    scales = qt.scales.unsqueeze(-1).to(torch.float32)
    out = (blocked * scales).reshape(bsz, heads, num_blocks * qt.block_size, head_dim)
    out = out[:, :, :seq_len, :].contiguous()
    if dtype is None:
        dtype = getattr(torch, qt.dtype_name, torch.float32)
    return out.to(dtype)


@dataclass
class QuantizedKVCache:
    layers: List[Tuple[QuantizedTensor, QuantizedTensor]]
    block_size: int = 16

    @classmethod
    def from_past_key_values(cls, past_key_values: PastKeyValues, block_size: int = 16) -> "QuantizedKVCache":
        layers = []
        for key, value in iter_kv_layers(past_key_values):
            layers.append((quantize_tensor_per_block(key, block_size), quantize_tensor_per_block(value, block_size)))
        return cls(layers=layers, block_size=block_size)

    def to_past_key_values(self, dtype=None):
        return tuple((dequantize_tensor(key, dtype), dequantize_tensor(value, dtype)) for key, value in self.layers)

    def memory_bytes(self) -> int:
        return sum(key.memory_bytes() + value.memory_bytes() for key, value in self.layers)


def floating_kv_memory_bytes(past_key_values: PastKeyValues, assume_fp16: bool = True) -> int:
    """Estimate native KV cache memory.

    The project compares against FP16 KV cache memory, so `assume_fp16=True`
    reports 2 bytes per element even when a CPU smoke run uses float32.
    """

    bytes_per_element = 2 if assume_fp16 else None
    total = 0
    for key, value in iter_kv_layers(past_key_values):
        if bytes_per_element is None:
            total += key.numel() * key.element_size() + value.numel() * value.element_size()
        else:
            total += (key.numel() + value.numel()) * bytes_per_element
    return total
