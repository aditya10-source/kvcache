from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

try:
    import torch
except ImportError:
    torch = None


@dataclass
class SimilarityMetrics:
    token_match_rate: float
    mean_abs_logit_error: float
    cosine_similarity: float
    mean_squared_error: float = 0.0
    kl_divergence: float = 0.0
    next_token_agreement: float = 0.0
    generated_text_exact_match: float = 0.0
    generated_text_overlap: float = 0.0
    edit_distance: int = 0
    per_step: List[dict] = field(default_factory=list)

    def as_dict(self):
        return {
            "token_match_rate": self.token_match_rate,
            "mean_abs_logit_error": self.mean_abs_logit_error,
            "cosine_similarity": self.cosine_similarity,
            "mean_squared_error": self.mean_squared_error,
            "kl_divergence": self.kl_divergence,
            "next_token_agreement": self.next_token_agreement,
            "generated_text_exact_match": self.generated_text_exact_match,
            "generated_text_overlap": self.generated_text_overlap,
            "edit_distance": self.edit_distance,
        }


def _edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (item_a != item_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _token_overlap(base, cand):
    if not base.numel() and not cand.numel():
        return 1.0
    base_set = set(base.tolist())
    cand_set = set(cand.tolist())
    denom = len(base_set | cand_set)
    return len(base_set & cand_set) / denom if denom else 0.0


def compare_outputs(baseline_tokens, candidate_tokens, baseline_logits: Sequence, candidate_logits: Sequence) -> SimilarityMetrics:
    if torch is None:
        raise RuntimeError("PyTorch is required. Install with `pip install -r requirements.txt`.")

    base = baseline_tokens.reshape(-1).cpu()
    cand = candidate_tokens.reshape(-1).cpu()
    n = min(base.numel(), cand.numel())
    token_match = (base[:n] == cand[:n]).float().mean().item() if n else 0.0
    exact = float(base.numel() == cand.numel() and bool(torch.equal(base, cand)))
    overlap = _token_overlap(base, cand)
    edit_distance = _edit_distance(base.tolist(), cand.tolist())

    per_step = []
    cosine_values = []
    mae_values = []
    mse_values = []
    kl_values = []
    argmax_matches = []

    for step, (b_logit, c_logit) in enumerate(zip(baseline_logits, candidate_logits)):
        b = b_logit.float().cpu()
        c = c_logit.float().cpu()
        b_flat = b.reshape(1, -1)
        c_flat = c.reshape(1, -1)
        cosine = torch.nn.functional.cosine_similarity(b_flat, c_flat).item()
        diff = b - c
        mae = diff.abs().mean().item()
        mse = diff.pow(2).mean().item()
        b_log_prob = torch.nn.functional.log_softmax(b, dim=-1)
        c_log_prob = torch.nn.functional.log_softmax(c, dim=-1)
        b_prob = b_log_prob.exp()
        kl = torch.nn.functional.kl_div(c_log_prob, b_prob, reduction="batchmean", log_target=False).item()
        b_argmax = int(torch.argmax(b, dim=-1).reshape(-1)[0].item())
        c_argmax = int(torch.argmax(c, dim=-1).reshape(-1)[0].item())
        match = int(b_argmax == c_argmax)
        cosine_values.append(cosine)
        mae_values.append(mae)
        mse_values.append(mse)
        kl_values.append(kl)
        argmax_matches.append(match)
        per_step.append(
            {
                "step": step,
                "cosine_similarity": cosine,
                "mae": mae,
                "mse": mse,
                "kl_divergence": kl,
                "baseline_argmax_token": b_argmax,
                "test_argmax_token": c_argmax,
                "argmax_match": match,
            }
        )

    def avg(values):
        return float(sum(values) / len(values)) if values else 0.0

    return SimilarityMetrics(
        token_match_rate=token_match,
        mean_abs_logit_error=avg(mae_values),
        cosine_similarity=avg(cosine_values),
        mean_squared_error=avg(mse_values),
        kl_divergence=avg(kl_values),
        next_token_agreement=avg(argmax_matches),
        generated_text_exact_match=exact,
        generated_text_overlap=overlap,
        edit_distance=edit_distance,
        per_step=per_step,
    )
