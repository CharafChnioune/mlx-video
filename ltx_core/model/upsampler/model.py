"""MLX LatentUpsampler wrapper."""

from mlx_video.models.ltx.upsampler import LatentUpsampler


def upsample_video(latent, video_encoder, upsampler: LatentUpsampler):
    """Compatibility shim: run upsampler on latent."""
    return upsampler(latent)


__all__ = ["LatentUpsampler", "upsample_video"]
