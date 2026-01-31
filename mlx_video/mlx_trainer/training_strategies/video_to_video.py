from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from .base_strategy import DEFAULT_FPS, ModelInputs, TrainingStrategy
from mlx_video.models.ltx.transformer import Modality


@dataclass
class VideoToVideoConfig:
    name: str = "video_to_video"
    first_frame_conditioning_p: float = 0.1


class VideoToVideoStrategy(TrainingStrategy):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.cfg.reference_latents_dir: "ref_latents",
        }

    def prepare_training_inputs(self, batch: dict[str, Any], timestep_sampler) -> ModelInputs:
        if hasattr(batch, "latents"):
            lat = batch.latents
            ref = batch.ref_latents or batch.__dict__.get("reference_latents")
            cond = batch.conditions
        else:
            lat = batch["latents"]
            ref = batch.get("reference_latents") or batch.get("ref_latents")
            cond = batch["conditions"]
        if ref is None:
            raise ValueError("video_to_video strategy requires reference_latents")

        target_latents = mx.array(lat["latents"], dtype=mx.float32)
        ref_latents = mx.array(ref["latents"], dtype=mx.float32)

        num_frames = int(np.array(lat["num_frames"]).reshape(-1)[0])
        height = int(np.array(lat["height"]).reshape(-1)[0])
        width = int(np.array(lat["width"]).reshape(-1)[0])
        fps = float(np.array(lat.get("fps", np.array([DEFAULT_FPS], dtype=np.float32))).reshape(-1)[0])

        ref_frames = int(np.array(ref["num_frames"]).reshape(-1)[0])
        ref_height = int(np.array(ref["height"]).reshape(-1)[0])
        ref_width = int(np.array(ref["width"]).reshape(-1)[0])

        target_latents = self._patchify_video(target_latents)
        ref_latents = self._patchify_video(ref_latents)

        b, target_seq_len, _ = target_latents.shape
        ref_seq_len = ref_latents.shape[1]

        raw_video_embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
        if raw_video_embeds is None:
            raise ValueError("Missing prompt embeddings in conditions")
        video_prompt_embeds = mx.array(raw_video_embeds, dtype=mx.float32)
        if video_prompt_embeds.ndim == 2:
            video_prompt_embeds = mx.expand_dims(video_prompt_embeds, axis=0)

        prompt_mask_np = cond.get("prompt_attention_mask")
        if prompt_mask_np is None:
            prompt_mask_np = np.ones((video_prompt_embeds.shape[1],), dtype=bool)
        prompt_mask = mx.array(prompt_mask_np)
        if prompt_mask.ndim == 1:
            prompt_mask = mx.expand_dims(prompt_mask, axis=0)

        ref_conditioning = mx.ones((b, ref_seq_len), dtype=mx.bool_)
        target_conditioning = self._create_first_frame_conditioning_mask(
            b, num_frames, height, width, p=self.cfg.first_frame_conditioning_p
        )
        conditioning_mask = mx.concatenate([ref_conditioning, target_conditioning], axis=1)

        sigmas = timestep_sampler.sample_for(target_latents, seq_len=ref_seq_len + target_seq_len)
        noise = mx.random.normal(target_latents.shape)
        sigmas_expanded = mx.reshape(sigmas, (b, 1, 1))
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise
        noisy_target = mx.where(mx.expand_dims(target_conditioning, -1), target_latents, noisy_target)
        targets = noise - target_latents

        combined_latents = mx.concatenate([ref_latents, noisy_target], axis=1)
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        ref_positions = self._get_video_positions(b, ref_frames, ref_height, ref_width, fps)
        target_positions = self._get_video_positions(b, num_frames, height, width, fps)
        positions = mx.concatenate([ref_positions, target_positions], axis=2)

        video_modality = Modality(
            latent=combined_latents,
            timesteps=timesteps.astype(mx.float32),
            positions=positions.astype(mx.float32),
            context=video_prompt_embeds,
            context_mask=prompt_mask,
            enabled=True,
        )

        ref_loss_mask = mx.zeros((b, ref_seq_len), dtype=mx.bool_)
        target_loss_mask = ~target_conditioning
        loss_mask = mx.concatenate([ref_loss_mask, target_loss_mask], axis=1)

        # Pad targets to align with combined sequence length
        ref_targets = mx.zeros((b, ref_seq_len, targets.shape[-1]), dtype=targets.dtype)
        combined_targets = mx.concatenate([ref_targets, targets], axis=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=combined_targets,
            audio_targets=None,
            video_loss_mask=loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
        )
