"""MLX model builder stub for compatibility with PyTorch loader API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_video.models.ltx import LTXModel
from mlx_video.models.ltx.video_vae import load_vae_decoder, load_vae_encoder
from mlx_video.models.ltx.audio_vae import AudioDecoder, Vocoder
from mlx_video.models.ltx.config import LTXModelConfig


class SingleGPUModelBuilder:
    def __init__(self, model_path: str, model_class_configurator: Any, model_sd_ops: Any = None):
        self.model_path = Path(model_path)
        self.model_class_configurator = model_class_configurator
        self.model_sd_ops = model_sd_ops

    def build(self, device: Any = None, dtype: mx.Dtype = mx.float16):
        # Heuristic: if configurator name suggests VAE, load VAE; else transformer.
        name = getattr(self.model_class_configurator, "__name__", str(self.model_class_configurator)).lower()
        if "encoder" in name and "video" in name:
            return load_vae_encoder(self.model_path, dtype=dtype)
        if "decoder" in name and "video" in name:
            return load_vae_decoder(self.model_path, dtype=dtype)
        if "audio" in name and "vocoder" not in name:
            return AudioDecoder.from_pretrained(self.model_path)
        if "vocoder" in name:
            return Vocoder.from_pretrained(self.model_path)

        # default: transformer
        config = LTXModelConfig()
        return LTXModel.from_pretrained(self.model_path, config=config, strict=False)
