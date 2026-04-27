from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


def get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for this command. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return torch


def select_device(preferred: Optional[str] = None):
    torch = get_torch()
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype_for_device(device):
    torch = get_torch()
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    torch = get_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device) -> None:
    torch = get_torch()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@dataclass
class TimerResult:
    seconds: float


def tensor_bytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()
