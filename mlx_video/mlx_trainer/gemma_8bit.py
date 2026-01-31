"""Legacy Gemma loader shim for MLX.

MLX uses `mlx_vlm` Gemma3 text encoder; this module provides a minimal
compatibility API for trainer configs that reference gemma_8bit.
"""

from __future__ import annotations

from pathlib import Path

from mlx_video.models.ltx.text_encoder import LTX2TextEncoder


def load_gemma(model_path: str | Path) -> LTX2TextEncoder:
    encoder = LTX2TextEncoder()
    encoder.load(model_path=Path(model_path), text_encoder_path=Path(model_path))
    return encoder

