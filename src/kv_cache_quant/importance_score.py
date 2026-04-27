from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

try:
    import torch
except ImportError:
    torch = None


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required. Install with `pip install -r requirements.txt`.")


def _normalize(scores):
    if scores.numel() == 0:
        return scores
    max_value = scores.max()
    min_value = scores.min()
    denom = (max_value - min_value).clamp(min=1e-8)
    return (scores - min_value) / denom


def recency_importance(num_blocks: int, device=None):
    """Return block scores where newer sequence blocks are more important."""

    _require_torch()
    if num_blocks <= 0:
        return torch.empty(0, device=device)
    scores = torch.arange(num_blocks, dtype=torch.float32, device=device)
    return _normalize(scores)


def attention_importance(attentions: Optional[Sequence], seq_len: int, block_size: int, device=None):
    """Aggregate attention mass into per-KV-block scores.

    `attentions` is expected to be the tuple returned by Hugging Face models
    when `output_attentions=True`; each tensor usually has shape
    `[B, H, Q, K]`. For decoding, Q is normally 1. For prefill, this uses
    the final query position because it best matches the next-token decision.
    """

    _require_torch()
    num_blocks = (seq_len + block_size - 1) // block_size
    if num_blocks <= 0:
        return torch.empty(0, device=device)
    if not attentions:
        return torch.zeros(num_blocks, dtype=torch.float32, device=device)

    block_scores = torch.zeros(num_blocks, dtype=torch.float32, device=device)
    used_layers = 0
    for attn in attentions:
        if attn is None or not torch.is_tensor(attn) or attn.ndim < 4:
            continue
        # [B, H, K] for the final query token.
        final_query = attn[..., -1, :seq_len].detach().float()
        if device is not None:
            final_query = final_query.to(device)
        token_scores = final_query.mean(dim=tuple(range(final_query.ndim - 1)))
        pad = num_blocks * block_size - token_scores.numel()
        if pad > 0:
            token_scores = torch.nn.functional.pad(token_scores, (0, pad))
        block_scores += token_scores.reshape(num_blocks, block_size).mean(dim=-1)
        used_layers += 1

    if used_layers:
        block_scores = block_scores / used_layers
    return _normalize(block_scores)


@dataclass
class ImportanceScores:
    recency: object
    attention: object
    hybrid: object


def compute_importance_scores(
    seq_len: int,
    block_size: int,
    attentions: Optional[Sequence] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    device=None,
) -> ImportanceScores:
    _require_torch()
    num_blocks = (seq_len + block_size - 1) // block_size
    recency = recency_importance(num_blocks, device=device)
    attention = attention_importance(attentions, seq_len, block_size, device=device)
    hybrid = _normalize(alpha * recency + beta * attention)
    return ImportanceScores(recency=recency, attention=attention, hybrid=hybrid)
