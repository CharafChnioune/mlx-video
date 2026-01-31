"""MLX latent upsampler wrappers."""

from mlx_video.models.ltx.upsampler import LatentUpsampler  # type: ignore


class LatentUpsamplerConfigurator:
    def __init__(self, *_, **__):
        pass


__all__ = ["LatentUpsampler", "LatentUpsamplerConfigurator"]
