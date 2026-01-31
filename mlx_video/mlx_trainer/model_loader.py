"""MLX model loader utilities for trainer parity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx

from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
from mlx_video.models.ltx.audio_vae import AudioDecoder, AudioEncoder, Vocoder, CausalityAxis, NormType
from mlx_video.generate import load_vae_encoder, load_vae_decoder
from mlx_video.convert import (
    load_audio_vae_weights,
    load_vocoder_weights,
    sanitize_audio_vae_weights,
    sanitize_vocoder_weights,
)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_transformer(checkpoint_path: str | Path, config: Optional[LTXModelConfig] = None) -> LTXModel:
    ckpt = _resolve_path(checkpoint_path)
    if config is None:
        config = LTXModelConfig(
            model_type=LTXModelType.VideoOnly,
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
    return LTXModel.from_pretrained([ckpt], config=config, strict=False)


def load_video_vae_encoder(checkpoint_path: str | Path):
    return load_vae_encoder(str(_resolve_path(checkpoint_path)))


def load_video_vae_decoder(checkpoint_path: str | Path):
    return load_vae_decoder(str(_resolve_path(checkpoint_path)), timestep_conditioning=None)


def load_audio_vae_decoder(checkpoint_path: str | Path):
    model_path = _resolve_path(checkpoint_path)
    root = model_path.parent if model_path.is_file() else model_path
    cfg_path = root / "audio_vae" / "config.json"
    cfg = {}
    if cfg_path.exists():
        import json

        cfg = json.loads(cfg_path.read_text())

    ch = int(cfg.get("base_channels", 128))
    ch_mult = tuple(cfg.get("ch_mult", (1, 2, 4)))
    num_res_blocks = int(cfg.get("num_res_blocks", 2))
    attn_resolutions = set(cfg.get("attn_resolutions") or [])
    resolution = int(cfg.get("resolution", 256))
    z_channels = int(cfg.get("latent_channels", 8))
    dropout = float(cfg.get("dropout", 0.0))
    norm_type = NormType(str(cfg.get("norm_type", "pixel")))
    causality_axis = CausalityAxis(str(cfg.get("causality_axis", "height")))
    mid_block_add_attention = bool(cfg.get("mid_block_add_attention", True))
    mel_bins = int(cfg.get("mel_bins", 64))
    sample_rate = int(cfg.get("sample_rate", 24000))
    mel_hop_length = int(cfg.get("mel_hop_length", 160))

    decoder = AudioDecoder(
        ch=ch,
        out_ch=2,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        resolution=resolution,
        z_channels=z_channels,
        norm_type=norm_type,
        causality_axis=causality_axis,
        dropout=dropout,
        mid_block_add_attention=mid_block_add_attention,
        sample_rate=sample_rate,
        mel_hop_length=mel_hop_length,
        mel_bins=mel_bins,
    )

    weights = load_audio_vae_weights(root)
    sanitized = sanitize_audio_vae_weights(weights)
    dec_weights = {k.replace("decoder.", ""): v for k, v in sanitized.items() if k.startswith("decoder.")}
    if dec_weights:
        decoder.load_weights(list(dec_weights.items()), strict=False)
    if "per_channel_statistics._mean_of_means" in sanitized:
        decoder.per_channel_statistics._mean_of_means = sanitized["per_channel_statistics._mean_of_means"]
    if "per_channel_statistics._std_of_means" in sanitized:
        decoder.per_channel_statistics._std_of_means = sanitized["per_channel_statistics._std_of_means"]
    return decoder


def load_vocoder(checkpoint_path: str | Path):
    model_path = _resolve_path(checkpoint_path)
    root = model_path.parent if model_path.is_file() else model_path
    cfg_path = root / "vocoder" / "config.json"
    cfg = {}
    if cfg_path.exists():
        import json

        cfg = json.loads(cfg_path.read_text())
    vocoder = Vocoder(
        resblock_kernel_sizes=cfg.get("resnet_kernel_sizes", [3, 7, 11]),
        upsample_rates=cfg.get("upsample_factors", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        resblock_dilation_sizes=cfg.get("resnet_dilations", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        upsample_initial_channel=cfg.get("hidden_channels", 1024),
        stereo=True,
        output_sample_rate=cfg.get("output_sampling_rate", 24000),
    )
    weights = load_vocoder_weights(root)
    sanitized = sanitize_vocoder_weights(weights)
    if sanitized:
        vocoder.load_weights(list(sanitized.items()), strict=False)
    return vocoder


def load_text_encoder(checkpoint_path: str | Path, text_encoder_path: str | Path) -> LTX2TextEncoder:
    model_path = _resolve_path(checkpoint_path)
    text_encoder_root = _resolve_path(text_encoder_path)
    encoder = LTX2TextEncoder()
    encoder.load(model_path=model_path, text_encoder_path=text_encoder_root)
    mx.eval(encoder.parameters())
    return encoder


@dataclass
class MLXModelComponents:
    transformer: LTXModel
    video_vae_encoder: object
    video_vae_decoder: object
    audio_decoder: object | None
    vocoder: object | None
    text_encoder: LTX2TextEncoder | None


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    with_audio: bool = False,
) -> MLXModelComponents:
    ckpt = _resolve_path(checkpoint_path)
    transformer = load_transformer(ckpt)
    video_vae_encoder = load_video_vae_encoder(ckpt)
    video_vae_decoder = load_video_vae_decoder(ckpt)
    audio_decoder = load_audio_vae_decoder(ckpt) if with_audio else None
    vocoder = load_vocoder(ckpt) if with_audio else None
    text_encoder = load_text_encoder(ckpt, text_encoder_path) if text_encoder_path else None
    return MLXModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
    )
