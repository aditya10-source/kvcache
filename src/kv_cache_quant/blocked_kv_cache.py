from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    import torch
except ImportError:
    torch = None

from .kv_quant import PastKeyValues, QuantizedTensor, dequantize_tensor, iter_kv_layers, quantize_tensor_per_block


@dataclass
class BlockedQuantizedTensor:
    values: "torch.Tensor"
    scales: "torch.Tensor"
    original_shape: Tuple[int, ...]
    block_size: int
    dtype_name: str

    def memory_bytes(self) -> int:
        return self.values.numel() * self.values.element_size() + self.scales.numel() * self.scales.element_size()


def block_quantized_tensor(qt: QuantizedTensor) -> BlockedQuantizedTensor:
    if torch is None:
        raise RuntimeError("PyTorch is required. Install with `pip install -r requirements.txt`.")
    bsz, heads, seq_len, head_dim = qt.original_shape
    pad = (qt.block_size - seq_len % qt.block_size) % qt.block_size
    values = qt.values
    if pad:
        values = torch.nn.functional.pad(values, (0, 0, 0, pad))
    num_blocks = values.shape[2] // qt.block_size
    blocked_values = values.reshape(bsz, heads, num_blocks, qt.block_size, head_dim).contiguous()
    return BlockedQuantizedTensor(
        values=blocked_values,
        scales=qt.scales.contiguous(),
        original_shape=qt.original_shape,
        block_size=qt.block_size,
        dtype_name=qt.dtype_name,
    )


def unblock_quantized_tensor(bqt: BlockedQuantizedTensor) -> QuantizedTensor:
    bsz, heads, seq_len, head_dim = bqt.original_shape
    flat = bqt.values.reshape(bsz, heads, -1, head_dim)[:, :, :seq_len, :].contiguous()
    return QuantizedTensor(
        values=flat,
        scales=bqt.scales,
        original_shape=bqt.original_shape,
        block_size=bqt.block_size,
        dtype_name=bqt.dtype_name,
    )


@dataclass
class BlockedQuantizedKVCache:
    layers: List[Tuple[BlockedQuantizedTensor, BlockedQuantizedTensor]]
    block_size: int = 16

    @classmethod
    def from_past_key_values(cls, past_key_values: PastKeyValues, block_size: int = 16) -> "BlockedQuantizedKVCache":
        layers = []
        for key, value in iter_kv_layers(past_key_values):
            q_key = quantize_tensor_per_block(key, block_size)
            q_value = quantize_tensor_per_block(value, block_size)
            layers.append((block_quantized_tensor(q_key), block_quantized_tensor(q_value)))
        return cls(layers=layers, block_size=block_size)

    def to_past_key_values(self, dtype=None):
        return tuple(
            (dequantize_tensor(unblock_quantized_tensor(key), dtype), dequantize_tensor(unblock_quantized_tensor(value), dtype))
            for key, value in self.layers
        )

    def memory_bytes(self) -> int:
        return sum(key.memory_bytes() + value.memory_bytes() for key, value in self.layers)
