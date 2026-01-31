"""Compatibility stub for LoRA fusion (handled in mlx_video.lora)."""

from __future__ import annotations

from typing import Any


def fuse_loras(*_args: Any, **_kwargs: Any):
    raise NotImplementedError("LoRA fusion is handled in mlx_video.lora for MLX models.")
