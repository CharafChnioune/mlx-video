from __future__ import annotations

from pathlib import Path
import json

from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.model.audio_vae import AudioDecoder, Vocoder
from ltx_core.model.transformer import X0Model
from ltx_core.model.upsampler import LatentUpsampler
from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
from ltx_core.text_encoders.gemma import AVGemmaTextEncoderModel
from mlx_video.generate import load_upsampler
from mlx_video.mlx_trainer import model_loader as mlx_model_loader
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.utils import get_model_path


class ModelLedger:
    """
    MLX-native model ledger that mirrors the legacy pipeline API but loads MLX models.

    This class is a lightweight compatibility layer so older pipeline code can
    run on Apple Silicon without CUDA.
    """

    def __init__(
        self,
        dtype,
        device: str,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        fp8transformer: bool = False,
        with_audio: bool = False,
    ):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.fp8transformer = fp8transformer
        self.with_audio = with_audio

        self.model_path = get_model_path(checkpoint_path) if checkpoint_path else None

    def _resolve_weight_file(self) -> Path:
        if self.model_path is None:
            raise ValueError("Model path not initialized.")
        if self.model_path.is_file():
            return self.model_path

        candidates = [
            "ltx-2-19b-dev-mlx.safetensors",
            "ltx-2-19b-distilled-mlx.safetensors",
            "ltx-2-19b-dev.safetensors",
            "ltx-2-19b-distilled.safetensors",
        ]
        for name in candidates:
            path = self.model_path / name
            if path.exists():
                return path

        # Fallback to first safetensors file
        for path in self.model_path.glob("*.safetensors"):
            return path

        raise FileNotFoundError(f"No safetensors weights found in {self.model_path}")

    def _load_config(self) -> LTXModelConfig:
        if self.model_path is None:
            raise ValueError("Model path not initialized.")
        cfg_path = self.model_path / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                return LTXModelConfig.from_dict(cfg)
            except Exception:
                pass

        config_kwargs = dict(
            model_type=LTXModelType.AudioVideo if self.with_audio else LTXModelType.VideoOnly,
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            caption_channels=3840,
            rope_type=LTXRopeType.SPLIT,
            double_precision_rope=True,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
        )
        if self.with_audio:
            config_kwargs.update(
                audio_num_attention_heads=32,
                audio_attention_head_dim=64,
                audio_in_channels=128 * 16,
                audio_out_channels=128 * 16,
                audio_cross_attention_dim=2048,
                audio_positional_embedding_max_pos=[20],
            )
        return LTXModelConfig(**config_kwargs)

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            fp8transformer=self.fp8transformer,
            with_audio=self.with_audio,
        )

    def transformer(self) -> X0Model:
        weight_file = self._resolve_weight_file()
        config = self._load_config()
        return mlx_model_loader.load_transformer(weight_file, config=config)

    def video_decoder(self) -> VideoDecoder:
        weight_file = self._resolve_weight_file()
        return mlx_model_loader.load_video_vae_decoder(weight_file)

    def video_encoder(self) -> VideoEncoder:
        weight_file = self._resolve_weight_file()
        return mlx_model_loader.load_video_vae_encoder(weight_file)

    def text_encoder(self) -> AVGemmaTextEncoderModel:
        if self.gemma_root_path is None:
            raise ValueError("Text encoder not initialized. Provide gemma_root_path.")
        weight_file = self._resolve_weight_file()
        return mlx_model_loader.load_text_encoder(weight_file, self.gemma_root_path)

    def audio_decoder(self) -> AudioDecoder:
        weight_file = self._resolve_weight_file()
        return mlx_model_loader.load_audio_vae_decoder(weight_file)

    def vocoder(self) -> Vocoder:
        weight_file = self._resolve_weight_file()
        return mlx_model_loader.load_vocoder(weight_file)

    def spatial_upsampler(self) -> LatentUpsampler:
        if self.spatial_upsampler_path is None:
            if self.model_path is None:
                raise ValueError("Upsampler not initialized.")
            path = self.model_path / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
        else:
            path = Path(self.spatial_upsampler_path)
        return load_upsampler(str(path))
