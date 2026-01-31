"""Compatibility shim for timestep embedding utilities (MLX)."""

from mlx_video.utils import get_timestep_embedding as PixArtAlphaCombinedTimestepSizeEmbeddings

__all__ = ["PixArtAlphaCombinedTimestepSizeEmbeddings"]
