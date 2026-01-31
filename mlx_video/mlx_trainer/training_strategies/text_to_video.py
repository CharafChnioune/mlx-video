from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from .base_strategy import DEFAULT_FPS, ModelInputs, TrainingStrategy
from mlx_video.models.ltx.transformer import Modality


@dataclass
class TextToVideoConfig:
    name: str = "text_to_video"
    first_frame_conditioning_p: float = 0.1
    with_audio: bool = False
    audio_latents_dir: str = "audio_latents"


class TextToVideoStrategy(TrainingStrategy):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    @property
    def requires_audio(self) -> bool:
        return bool(self.cfg.with_audio)

    def get_data_sources(self) -> dict[str, str]:
        sources = {"latents": "latents", "conditions": "conditions"}
        if self.cfg.with_audio:
            sources[self.cfg.audio_latents_dir] = "audio_latents"
        return sources

    def prepare_training_inputs(self, batch: dict[str, Any], timestep_sampler) -> ModelInputs:
        if hasattr(batch, "latents"):
            lat = batch.latents
            cond = batch.conditions
            audio_latents = batch.audio_latents
        else:
            lat = batch["latents"]
            cond = batch["conditions"]
            audio_latents = batch.get("audio_latents")
        video_latents = mx.array(lat["latents"], dtype=mx.float32)
        num_frames = int(np.array(lat["num_frames"]).reshape(-1)[0])
        height = int(np.array(lat["height"]).reshape(-1)[0])
        width = int(np.array(lat["width"]).reshape(-1)[0])
        fps = float(np.array(lat.get("fps", np.array([DEFAULT_FPS], dtype=np.float32))).reshape(-1)[0])

        video_latents = self._patchify_video(video_latents)
        b, seq_len, _ = video_latents.shape

        raw_video_embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
        if raw_video_embeds is None:
            raise ValueError("Missing prompt embeddings in conditions")
        video_prompt_embeds = mx.array(raw_video_embeds, dtype=mx.float32)
        audio_prompt_embeds = mx.array(cond.get("audio_prompt_embeds", raw_video_embeds), dtype=mx.float32)

        if video_prompt_embeds.ndim == 2:
            video_prompt_embeds = mx.expand_dims(video_prompt_embeds, axis=0)
        if audio_prompt_embeds.ndim == 2:
            audio_prompt_embeds = mx.expand_dims(audio_prompt_embeds, axis=0)

        prompt_mask_np = cond.get("prompt_attention_mask")
        if prompt_mask_np is None:
            prompt_mask_np = np.ones((video_prompt_embeds.shape[1],), dtype=bool)
        prompt_mask = mx.array(prompt_mask_np)
        if prompt_mask.ndim == 1:
            prompt_mask = mx.expand_dims(prompt_mask, axis=0)

        conditioning_mask = self._create_first_frame_conditioning_mask(
            b, num_frames, height, width, p=self.cfg.first_frame_conditioning_p
        )

        sigmas = timestep_sampler.sample_for(video_latents, seq_len=seq_len)
        noise = mx.random.normal(video_latents.shape)
        sigmas_expanded = mx.reshape(sigmas, (b, 1, 1))
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * noise
        noisy_video = mx.where(mx.expand_dims(conditioning_mask, -1), video_latents, noisy_video)

        video_targets = noise - video_latents
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())
        positions = self._get_video_positions(b, num_frames, height, width, fps)

        video_modality = Modality(
            latent=noisy_video,
            timesteps=timesteps.astype(mx.float32),
            positions=positions.astype(mx.float32),
            context=video_prompt_embeds,
            context_mask=prompt_mask,
            enabled=True,
        )

        video_loss_mask = ~conditioning_mask

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None
        if self.cfg.with_audio and audio_latents is not None:
            a_lat = audio_latents
            audio_latents = mx.array(a_lat["latents"], dtype=mx.float32)
            audio_latents = self._patchify_audio(audio_latents)
            ab, at, _ = audio_latents.shape
            audio_noise = mx.random.normal(audio_latents.shape)
            a_sigmas = mx.broadcast_to(mx.reshape(sigmas, (b, 1, 1)), audio_latents.shape)
            noisy_audio = (1 - a_sigmas) * audio_latents + a_sigmas * audio_noise
            audio_targets = audio_noise - audio_latents
            audio_timesteps = mx.broadcast_to(mx.reshape(sigmas.squeeze(), (b, 1)), (ab, at))
            audio_positions = self._get_audio_positions(ab, at)
            audio_modality = Modality(
                latent=noisy_audio,
                timesteps=audio_timesteps.astype(mx.float32),
                positions=audio_positions.astype(mx.float32),
                context=audio_prompt_embeds,
                context_mask=prompt_mask,
                enabled=True,
            )
            audio_loss_mask = mx.ones((ab, at), dtype=mx.bool_)

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )
