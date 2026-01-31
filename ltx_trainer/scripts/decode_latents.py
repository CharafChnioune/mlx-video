#!/usr/bin/env python3
"""Decode precomputed latents to video (and optional audio) using MLX VAE."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open

from mlx_video.generate import (
    load_vae_decoder,
    load_audio_decoder,
    load_vocoder,
    save_audio,
    mux_video_audio,
    _write_video_cv2,
    AUDIO_SAMPLE_RATE,
    PipelineType,
)


def _load_latents(path: Path, key: str | None = None) -> mx.array:
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="np") as f:
            if key and key in f.keys():
                data = f.get_tensor(key)
            elif "latents" in f.keys():
                data = f.get_tensor("latents")
            else:
                # fallback: first tensor
                key = next(iter(f.keys()))
                data = f.get_tensor(key)
        return mx.array(data)
    raise ValueError(f"Unsupported latent format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-repo", default="Lightricks/LTX-2")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--audio-latents", type=str, default=None, help="Optional audio latents (.safetensors)")
    parser.add_argument("--pipeline", type=str, default="dev", choices=["dev", "distilled"])
    parser.add_argument("--output-audio", type=str, default=None, help="Optional output wav path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    latents = _load_latents(Path(args.latents))
    if latents.ndim == 4:
        # (C, F, H, W) -> (1, C, F, H, W)
        latents = mx.expand_dims(latents, axis=0)
    vae = load_vae_decoder(args.model_repo, timestep_conditioning=None)
    video = vae(latents)
    mx.eval(video)

    video = mx.squeeze(video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video_path = output_path

    if args.audio_latents:
        temp_video_path = output_path.with_suffix(".temp.mp4")

    _write_video_cv2(video_np, temp_video_path, args.fps)

    # Optional audio decode + mux
    if args.audio_latents:
        audio_latents = _load_latents(Path(args.audio_latents))
        if audio_latents.ndim == 3:
            # (C, T, F) -> (1, C, T, F)
            audio_latents = mx.expand_dims(audio_latents, axis=0)
        if args.debug:
            print(f"[decode_latents] audio_latents shape={audio_latents.shape}")
        audio_decoder = load_audio_decoder(args.model_repo, PipelineType(args.pipeline))
        vocoder = load_vocoder(args.model_repo, PipelineType(args.pipeline))
        mel = audio_decoder(audio_latents)
        mx.eval(mel)
        audio_waveform = vocoder(mel)
        mx.eval(audio_waveform)
        audio_np = np.array(audio_waveform.astype(mx.float32))
        if audio_np.ndim == 3:
            audio_np = audio_np[0]
        audio_path = Path(args.output_audio) if args.output_audio else output_path.with_suffix(".wav")
        save_audio(audio_np, audio_path, AUDIO_SAMPLE_RATE)
        mux_ok = mux_video_audio(temp_video_path, audio_path, output_path)
        if mux_ok and temp_video_path.exists() and temp_video_path != output_path:
            temp_video_path.unlink()
        elif not mux_ok:
            print("[decode_latents] ⚠️ Audio mux failed; leaving video without audio.")


if __name__ == "__main__":
    main()
