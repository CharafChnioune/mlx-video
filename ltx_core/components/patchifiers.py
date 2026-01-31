from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx

from ltx_core.components.protocols import Patchifier
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape


class VideoLatentPatchifier(Patchifier):
    def __init__(self, patch_size: int):
        self._patch_size = (
            1,  # temporal
            patch_size,
            patch_size,
        )

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        return math.prod(tgt_shape.to_shape()[2:]) // math.prod(self._patch_size)

    def patchify(self, latents: mx.array) -> mx.array:
        b, c, f, h, w = latents.shape
        p1, p2, p3 = self._patch_size
        if f % p1 != 0 or h % p2 != 0 or w % p3 != 0:
            raise ValueError("Latents not divisible by patch size")
        latents = mx.reshape(latents, (b, c, f // p1, p1, h // p2, p2, w // p3, p3))
        latents = mx.transpose(latents, (0, 2, 4, 6, 1, 3, 5, 7))
        tokens = mx.reshape(latents, (b, (f // p1) * (h // p2) * (w // p3), c * p1 * p2 * p3))
        return tokens

    def unpatchify(self, latents: mx.array, output_shape: VideoLatentShape) -> mx.array:
        p1, p2, p3 = self._patch_size
        if p1 != 1:
            raise ValueError("Temporal patch size must be 1 for symmetric patchifier")
        f = output_shape.frames // p1
        h = output_shape.height // p2
        w = output_shape.width // p3
        b = output_shape.batch
        c = output_shape.channels
        latents = mx.reshape(latents, (b, f, h, w, c, p1, p2, p3))
        latents = mx.transpose(latents, (0, 4, 1, 5, 2, 6, 3, 7))
        return mx.reshape(latents, (b, c, output_shape.frames, output_shape.height, output_shape.width))

    def get_patch_grid_bounds(self, output_shape: AudioLatentShape | VideoLatentShape, device: Optional[str] = None) -> mx.array:
        if not isinstance(output_shape, VideoLatentShape):
            raise ValueError("VideoLatentPatchifier expects VideoLatentShape when computing coordinates")

        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch

        grid_f = mx.arange(0, frames, step=self._patch_size[0])
        grid_h = mx.arange(0, height, step=self._patch_size[1])
        grid_w = mx.arange(0, width, step=self._patch_size[2])
        grid_coords = mx.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        patch_starts = mx.stack(grid_coords, axis=0)
        patch_size_delta = mx.array(self._patch_size).reshape(3, 1, 1, 1)
        patch_ends = patch_starts + patch_size_delta
        latent_coords = mx.stack((patch_starts, patch_ends), axis=-1)
        latent_coords = mx.expand_dims(latent_coords, axis=0)
        latent_coords = mx.broadcast_to(latent_coords, (batch_size,) + latent_coords.shape[1:])
        latent_coords = mx.reshape(latent_coords, (batch_size, 3, -1, 2))
        return latent_coords


def get_pixel_coords(
    latent_coords: mx.array,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> mx.array:
    scale_tensor = mx.array(scale_factors).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_tensor
    if causal_fix:
        adjusted = mx.maximum(pixel_coords[:, 0, ...] + 1 - scale_factors.time, 0)
        pixel_coords = mx.concatenate(
            [mx.expand_dims(adjusted, axis=1), pixel_coords[:, 1:, ...]], axis=1
        )
    return pixel_coords


class AudioPatchifier(Patchifier):
    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
    ) -> mx.array:
        audio_latent_frame = mx.arange(start_latent, end_latent)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor
        if self.is_causal:
            audio_mel_frame = audio_mel_frame + 1
        return (audio_mel_frame * self.hop_length) / self.sample_rate

    def _compute_audio_timings(self, batch_size: int, num_steps: int) -> mx.array:
        start_timings = self._get_audio_latent_time_in_sec(self.shift, num_steps + self.shift)
        start_timings = mx.expand_dims(start_timings, axis=0)
        start_timings = mx.broadcast_to(start_timings, (batch_size, start_timings.shape[1]))
        start_timings = mx.expand_dims(start_timings, axis=1)

        end_timings = self._get_audio_latent_time_in_sec(self.shift + 1, num_steps + self.shift + 1)
        end_timings = mx.expand_dims(end_timings, axis=0)
        end_timings = mx.broadcast_to(end_timings, (batch_size, end_timings.shape[1]))
        end_timings = mx.expand_dims(end_timings, axis=1)

        return mx.stack([start_timings, end_timings], axis=-1)

    def patchify(self, audio_latents: mx.array) -> mx.array:
        b, c, t, f = audio_latents.shape
        return mx.reshape(mx.transpose(audio_latents, (0, 2, 1, 3)), (b, t, c * f))

    def unpatchify(self, audio_latents: mx.array, output_shape: AudioLatentShape) -> mx.array:
        b, t, cf = audio_latents.shape
        c = output_shape.channels
        f = output_shape.mel_bins
        return mx.transpose(mx.reshape(audio_latents, (b, t, c, f)), (0, 2, 1, 3))

    def get_patch_grid_bounds(self, output_shape: AudioLatentShape | VideoLatentShape, device: Optional[str] = None) -> mx.array:
        if not isinstance(output_shape, AudioLatentShape):
            raise ValueError("AudioPatchifier expects AudioLatentShape when computing coordinates")
        return self._compute_audio_timings(output_shape.batch, output_shape.frames)
