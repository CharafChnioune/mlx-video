from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx


class VideoPixelShape(NamedTuple):
    """Shape of decoded video pixels (B, F, H, W, fps)."""

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    """Downscaling factors from pixel space to latent space."""

    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """Latent tensor shape (B, C, F, H, W)."""

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_shape(self) -> tuple[int, int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.height, self.width)

    @staticmethod
    def from_shape(shape: tuple[int, ...]) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)

    @staticmethod
    def from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: int = 128,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors.time + 1
        height = shape.height // scale_factors.height
        width = shape.width // scale_factors.width

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    """Audio latent shape (B, C, T, mel_bins)."""

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.mel_bins)

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_shape(shape: tuple[int, ...]) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


@dataclass(frozen=True)
class LatentState:
    """Latent diffusion state (MLX)."""

    latent: mx.array
    denoise_mask: mx.array
    positions: mx.array
    clean_latent: mx.array

    def clone(self) -> "LatentState":
        return LatentState(
            latent=mx.array(self.latent),
            denoise_mask=mx.array(self.denoise_mask),
            positions=mx.array(self.positions),
            clean_latent=mx.array(self.clean_latent),
        )
