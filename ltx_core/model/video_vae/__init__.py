"""MLX video VAE wrappers."""

from mlx_video.models.ltx.video_vae import (  # type: ignore
    VideoEncoder,
    VideoDecoder,
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
)

# Minimal compat constants
VAE_ENCODER_COMFY_KEYS_FILTER: dict[str, str] = {}
VAE_DECODER_COMFY_KEYS_FILTER: dict[str, str] = {}


class VideoEncoderConfigurator:
    def __init__(self, *_, **__):
        pass


class VideoDecoderConfigurator:
    def __init__(self, *_, **__):
        pass


__all__ = [
    "VideoEncoder",
    "VideoDecoder",
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "VAE_ENCODER_COMFY_KEYS_FILTER",
    "VAE_DECODER_COMFY_KEYS_FILTER",
    "VideoEncoderConfigurator",
    "VideoDecoderConfigurator",
]
