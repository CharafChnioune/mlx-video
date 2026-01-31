"""Compatibility stub for SFT loader."""

from __future__ import annotations

from typing import Any


def SafetensorsModelStateDictLoader(*_args: Any, **_kwargs: Any):
    raise NotImplementedError("SFT loader is not implemented for MLX in this repo.")
