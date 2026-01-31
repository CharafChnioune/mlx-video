"""Compatibility wrapper for MLX transformer model."""

from mlx_video.models.ltx.ltx import LTXModel, X0Model  # re-export

__all__ = ["LTXModel", "X0Model"]
