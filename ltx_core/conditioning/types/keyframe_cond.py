from __future__ import annotations

import mlx.core as mx

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape


class VideoConditionByKeyframeIndex(ConditioningItem):
    """Conditions video generation on keyframe latents at a specific frame index."""

    def __init__(self, keyframes: mx.array, frame_idx: int, strength: float):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: VideoLatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_shape(self.keyframes.shape),
            device=None,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        )

        positions = positions.astype(mx.float32)
        positions = positions.at[:, 0, ...].set(positions[:, 0, ...] + self.frame_idx)
        positions = positions.at[:, 0, ...].set(positions[:, 0, ...] / latent_tools.fps)

        denoise_mask = mx.full(tokens.shape[:2] + (1,), 1.0 - self.strength, dtype=self.keyframes.dtype)

        return LatentState(
            latent=mx.concatenate([latent_state.latent, tokens], axis=1),
            denoise_mask=mx.concatenate([latent_state.denoise_mask, denoise_mask], axis=1),
            positions=mx.concatenate([latent_state.positions, positions], axis=2),
            clean_latent=mx.concatenate([latent_state.clean_latent, tokens], axis=1),
        )
