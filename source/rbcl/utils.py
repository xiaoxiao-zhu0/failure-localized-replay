"""Small utilities shared by RBCL experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def configure_determinism(enabled: bool) -> None:
    """Configure reproducible CUDA/CPU kernels for paired no-op controls."""
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_seed(seed: int) -> None:
    """Fix random seeds so budget comparisons use the same task/order noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cuda: int) -> torch.device:
    """Return the device requested by the experiment entry script."""
    if cuda >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda}")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create an output directory and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_jsonable(value: Any) -> Any:
    """Convert Avalanche/PyTorch metric values to JSON-safe objects."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write experiment metadata/results in a reusable format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)
