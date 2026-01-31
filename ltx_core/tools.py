from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import mlx.core as mx

from ltx_core.components.patchifiers import (
    AudioLatentShape,
    AudioPatchifier,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)
from ltx_core.components.protocols import Patchifier
from ltx_core.types import LatentState, SpatioTemporalScaleFactors

DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class LatentTools(Protocol):
    patchifier: Patchifier
    target_shape: VideoLatentShape | AudioLatentShape

    def create_initial_state(
        self,
        device: str,
        dtype: mx.Dtype,
        initial_latent: mx.array | None = None,
    ) -> LatentState: ...

    def patchify(self, latent_state: LatentState) -> LatentState:
        if latent_state.latent.shape != self.target_shape.to_shape():
            raise ValueError(
                f"Latent state has shape {latent_state.latent.shape}, expected shape is {self.target_shape.to_shape()}"
            )
        latent_state = latent_state.clone()
        latent = self.patchifier.patchify(latent_state.latent)
        clean_latent = self.patchifier.patchify(latent_state.clean_latent)
        denoise_mask = self.patchifier.patchify(latent_state.denoise_mask)
        return replace(latent_state, latent=latent, denoise_mask=denoise_mask, clean_latent=clean_latent)

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        latent_state = latent_state.clone()
        latent = self.patchifier.unpatchify(latent_state.latent, output_shape=self.target_shape)
        clean_latent = self.patchifier.unpatchify(latent_state.clean_latent, output_shape=self.target_shape)
        denoise_mask = self.patchifier.unpatchify(latent_state.denoise_mask, output_shape=self.target_shape.mask_shape())
        return replace(latent_state, latent=latent, denoise_mask=denoise_mask, clean_latent=clean_latent)

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        latent_state = latent_state.clone()
        num_tokens = self.patchifier.get_token_count(self.target_shape)
        latent = latent_state.latent[:, :num_tokens]
        clean_latent = latent_state.clean_latent[:, :num_tokens]
        denoise_mask = mx.ones_like(latent_state.denoise_mask)[:, :num_tokens]
        positions = latent_state.positions[:, :, :num_tokens]
        return LatentState(latent=latent, denoise_mask=denoise_mask, positions=positions, clean_latent=clean_latent)


@dataclass(frozen=True)
class VideoLatentTools(LatentTools):
    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    scale_factors: SpatioTemporalScaleFactors = DEFAULT_SCALE_FACTORS
    causal_fix: bool = True

    def create_initial_state(
        self,
        device: str,
        dtype: mx.Dtype,
        initial_latent: mx.array | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_shape()}"
            )
        else:
            initial_latent = mx.zeros(self.target_shape.to_shape(), dtype=dtype)

        clean_latent = mx.array(initial_latent)

        denoise_mask = mx.ones(self.target_shape.mask_shape().to_shape(), dtype=mx.float32)

        latent_coords = self.patchifier.get_patch_grid_bounds(output_shape=self.target_shape, device=device)

        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.scale_factors,
            causal_fix=self.causal_fix,
        ).astype(mx.float32)
        positions = positions.at[:, 0, ...].set(positions[:, 0, ...] / self.fps)

        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions.astype(dtype),
                clean_latent=clean_latent,
            )
        )


@dataclass(frozen=True)
class AudioLatentTools(LatentTools):
    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        device: str,
        dtype: mx.Dtype,
        initial_latent: mx.array | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_shape()}"
            )
        else:
            initial_latent = mx.zeros(self.target_shape.to_shape(), dtype=dtype)

        clean_latent = mx.array(initial_latent)

        denoise_mask = mx.ones(self.target_shape.mask_shape().to_shape(), dtype=mx.float32)

        latent_coords = self.patchifier.get_patch_grid_bounds(output_shape=self.target_shape, device=device)

        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=latent_coords,
                clean_latent=clean_latent,
            )
        )
