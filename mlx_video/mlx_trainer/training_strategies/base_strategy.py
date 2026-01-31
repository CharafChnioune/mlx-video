from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from mlx_video.generate import create_position_grid, create_audio_position_grid
from mlx_video.models.ltx.transformer import Modality


DEFAULT_FPS = 24.0


@dataclass
class ModelInputs:
    video: Modality
    audio: Modality | None
    video_targets: mx.array
    audio_targets: mx.array | None
    video_loss_mask: mx.array
    audio_loss_mask: mx.array | None
    ref_seq_len: int | None = None


class TrainingStrategy:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @property
    def requires_audio(self) -> bool:
        return False

    def get_data_sources(self) -> list[str] | dict[str, str]:
        raise NotImplementedError

    def prepare_training_inputs(self, batch: dict[str, Any], timestep_sampler) -> ModelInputs:
        raise NotImplementedError

    def compute_loss(self, video_pred: mx.array, audio_pred: mx.array | None, inputs: ModelInputs) -> mx.array:
        v = (video_pred - inputs.video_targets) ** 2
        v = mx.sum(v, axis=-1)
        v = mx.where(inputs.video_loss_mask, v, mx.zeros_like(v))
        v_loss = mx.sum(v) / mx.maximum(mx.sum(inputs.video_loss_mask), 1)

        a_loss = mx.array(0.0)
        if audio_pred is not None and inputs.audio_targets is not None and inputs.audio_loss_mask is not None:
            a = (audio_pred - inputs.audio_targets) ** 2
            a = mx.sum(a, axis=-1)
            a = mx.where(inputs.audio_loss_mask, a, mx.zeros_like(a))
            a_loss = mx.sum(a) / mx.maximum(mx.sum(inputs.audio_loss_mask), 1)
        return v_loss + a_loss

    @staticmethod
    def _patchify_video(latents: mx.array) -> mx.array:
        while latents.ndim > 5 and latents.shape[1] == 1:
            latents = mx.squeeze(latents, axis=1)
        b, c, f, h, w = latents.shape
        x = mx.transpose(latents, (0, 2, 3, 4, 1))
        return mx.reshape(x, (b, f * h * w, c))

    @staticmethod
    def _patchify_audio(latents: mx.array) -> mx.array:
        while latents.ndim > 4 and latents.shape[1] == 1:
            latents = mx.squeeze(latents, axis=1)
        b, c, t, f = latents.shape
        x = mx.transpose(latents, (0, 2, 1, 3))
        return mx.reshape(x, (b, t, c * f))

    @staticmethod
    def _create_first_frame_conditioning_mask(b: int, f: int, h: int, w: int, p: float) -> mx.array:
        if f <= 0:
            return mx.zeros((b, 0), dtype=mx.bool_)
        first = mx.ones((b, 1, h, w), dtype=mx.bool_)
        if f > 1:
            rest = mx.zeros((b, f - 1, h, w), dtype=mx.bool_)
            mask = mx.concatenate([first, rest], axis=1)
        else:
            mask = first
        mask = mx.reshape(mask, (b, f * h * w))
        if p <= 0:
            return mask * False
        if p >= 1:
            return mask
        keep = mx.random.uniform(shape=(b, 1)) < p
        return mx.where(keep, mask, mx.zeros_like(mask))

    @staticmethod
    def _create_per_token_timesteps(conditioning_mask: mx.array, sigmas: mx.array) -> mx.array:
        if sigmas.ndim == 2:
            sigmas = mx.squeeze(sigmas, axis=-1)
        expanded = mx.broadcast_to(mx.reshape(sigmas, (-1, 1)), conditioning_mask.shape)
        return mx.where(conditioning_mask, mx.zeros_like(expanded), expanded)

    @staticmethod
    def _get_video_positions(batch_size: int, num_frames: int, height: int, width: int, fps: float) -> mx.array:
        return create_position_grid(batch_size, num_frames, height, width, fps=fps)

    @staticmethod
    def _get_audio_positions(batch_size: int, num_steps: int) -> mx.array:
        return create_audio_position_grid(batch_size, num_steps)
