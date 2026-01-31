"""MLX memory helpers (CPU/GPU agnostic on Apple Silicon)."""

from __future__ import annotations

import gc
import time

import mlx.core as mx


def cleanup_memory() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def log_memory(tag: str = "") -> str:
    active = mx.get_active_memory() / (1024 ** 3)
    peak = mx.get_peak_memory() / (1024 ** 3)
    msg = f"[memory] {tag} active={active:.2f}GB peak={peak:.2f}GB"
    print(msg)
    return msg


def set_seed(seed: int) -> None:
    mx.random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def time_sync() -> float:
    return time.time()

