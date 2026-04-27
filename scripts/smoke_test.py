from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Smoke test needs torch. Install with `pip install -r requirements.txt`.") from exc

    from kv_cache_quant.adaptive_kv_policy import AdaptiveHybridKVCache, AdaptiveKVPolicy
    from kv_cache_quant.blocked_kv_cache import BlockedQuantizedKVCache
    from kv_cache_quant.kv_quant import QuantizedKVCache, floating_kv_memory_bytes

    torch.manual_seed(0)
    past = tuple(
        (
            torch.randn(1, 2, 17, 8, dtype=torch.float32),
            torch.randn(1, 2, 17, 8, dtype=torch.float32),
        )
        for _ in range(2)
    )

    q_cache = QuantizedKVCache.from_past_key_values(past, block_size=8)
    dq = q_cache.to_past_key_values(dtype=torch.float32)
    b_cache = BlockedQuantizedKVCache.from_past_key_values(past, block_size=8)
    bdq = b_cache.to_past_key_values(dtype=torch.float32)
    policy = AdaptiveKVPolicy(block_size=8, fp16_ratio=0.34, int8_ratio=0.34, importance_mode="recency")
    a_cache = AdaptiveHybridKVCache.from_past_key_values(past, policy=policy)
    adq = a_cache.to_past_key_values(dtype=torch.float32)

    assert dq[0][0].shape == past[0][0].shape
    assert bdq[0][1].shape == past[0][1].shape
    assert q_cache.memory_bytes() < floating_kv_memory_bytes(past, assume_fp16=False)
    assert b_cache.layers[0][0].values.shape == (1, 2, 3, 8, 8)
    assert adq[0][0].shape == past[0][0].shape
    assert a_cache.precision_counts()["fp16"] >= 1

    print("smoke_test passed")
    print(f"float32 cache bytes: {floating_kv_memory_bytes(past, assume_fp16=False)}")
    print(f"int8 cache bytes:    {q_cache.memory_bytes()}")
    print(f"blocked shape:       {tuple(b_cache.layers[0][0].values.shape)}")
    print(f"adaptive bytes:      {a_cache.memory_bytes()}")
    print(f"adaptive precision:  {a_cache.precision_counts()}")


if __name__ == "__main__":
    main()
