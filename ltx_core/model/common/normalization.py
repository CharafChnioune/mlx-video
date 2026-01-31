"""MLX normalization helpers."""

from enum import Enum

from mlx_video.utils import PixelNorm


class NormType(str, Enum):
    PIXEL_NORM = "pixel_norm"
    GROUP_NORM = "group_norm"
    LAYER_NORM = "layer_norm"


def build_normalization_layer(*_args, **_kwargs):
    return PixelNorm()


__all__ = ["NormType", "PixelNorm", "build_normalization_layer"]
