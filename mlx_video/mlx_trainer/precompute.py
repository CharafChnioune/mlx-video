from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
from mlx_video.models.ltx.audio_vae import AudioEncoder, AudioProcessor, CausalityAxis, NormType
from mlx_video.convert import load_audio_vae_weights, sanitize_audio_vae_weights
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.utils import get_model_path
from .video_utils import read_video
from .captioning import get_captioner


def _ensure_frames(frames: np.ndarray) -> np.ndarray:
    # Ensure 1 + 8*k frames by trimming or padding last frame
    f = frames.shape[0]
    if f < 1:
        raise ValueError("No frames")
    if (f - 1) % 8 == 0:
        return frames
    target = 1 + 8 * ((f - 1) // 8)
    if target < 1:
        target = 1
    if target == f:
        return frames
    if target < f:
        return frames[:target]
    # pad
    pad = target - f
    last = frames[-1:]
    pads = np.repeat(last, pad, axis=0)
    return np.concatenate([frames, pads], axis=0)


def _load_prompts(prompts_file: Optional[str]) -> Dict[str, str]:
    if not prompts_file:
        return {}
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    prompts: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if "|" in line:
            name, prompt = line.split("|", 1)
            prompts[name.strip()] = prompt.strip()
    return prompts


def _bucket_score(frames: int, height: int, width: int, target: Tuple[int, int, int]) -> float:
    target_f, target_h, target_w = target
    # Relative deltas to keep scales comparable across buckets
    f_term = abs(frames - target_f) / max(target_f, 1)
    h_term = abs(height - target_h) / max(target_h, 1)
    w_term = abs(width - target_w) / max(target_w, 1)
    return f_term + h_term + w_term


def _select_bucket(frames: np.ndarray, buckets: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    f, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    best = None
    best_score = None
    for bucket in buckets:
        score = _bucket_score(f, h, w, bucket)
        if best_score is None or score < best_score:
            best_score = score
            best = bucket
    if best is None:
        raise ValueError("No buckets provided")
    return best


def _match_frame_count(frames: np.ndarray, target_f: int) -> np.ndarray:
    frames = _ensure_frames(frames)
    if frames.shape[0] == target_f:
        return frames
    if frames.shape[0] > target_f:
        return frames[:target_f]
    pad = target_f - frames.shape[0]
    return np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)


def _resize_and_crop(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    import cv2

    h, w = frames.shape[1], frames.shape[2]
    scale = max(target_w / float(w), target_h / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames])
    start_x = max((new_w - target_w) // 2, 0)
    start_y = max((new_h - target_h) // 2, 0)
    return resized[:, start_y : start_y + target_h, start_x : start_x + target_w]


def encode_video_latents(frames: np.ndarray, encoder) -> np.ndarray:
    # frames: [F,H,W,3] in [0,1]
    frames = frames.astype(np.float32)
    frames = frames * 2.0 - 1.0
    frames = np.transpose(frames, (3, 0, 1, 2))
    frames = np.expand_dims(frames, axis=0)  # [B,C,F,H,W]
    latents = encoder(mx.array(frames))
    latents = np.array(latents)
    return latents


def _load_audio_vae_config(model_path: Path) -> Dict[str, object]:
    cfg_path = model_path / "audio_vae" / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _build_audio_encoder(model_path: Path) -> tuple[AudioEncoder, AudioProcessor]:
    cfg = _load_audio_vae_config(model_path)

    ch = int(cfg.get("base_channels", 128))
    ch_mult = tuple(cfg.get("ch_mult", (1, 2, 4)))
    num_res_blocks = int(cfg.get("num_res_blocks", 2))
    attn_resolutions = set(cfg.get("attn_resolutions") or [])
    resolution = int(cfg.get("resolution", 256))
    z_channels = int(cfg.get("latent_channels", 8))
    double_z = bool(cfg.get("double_z", True))
    dropout = float(cfg.get("dropout", 0.0))
    in_channels = int(cfg.get("in_channels", 2))
    norm_type = NormType(str(cfg.get("norm_type", "pixel")))
    causality_axis = CausalityAxis(str(cfg.get("causality_axis", "height")))
    mid_block_add_attention = bool(cfg.get("mid_block_add_attention", True))
    sample_rate = int(cfg.get("sample_rate", 16000))
    mel_hop_length = int(cfg.get("mel_hop_length", 160))
    mel_bins = int(cfg.get("mel_bins", 64))
    n_fft = int(cfg.get("n_fft", 1024))

    encoder = AudioEncoder(
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        resamp_with_conv=True,
        in_channels=in_channels,
        resolution=resolution,
        z_channels=z_channels,
        double_z=double_z,
        norm_type=norm_type,
        causality_axis=causality_axis,
        mid_block_add_attention=mid_block_add_attention,
        sample_rate=sample_rate,
        mel_hop_length=mel_hop_length,
        n_fft=n_fft,
        mel_bins=mel_bins,
        is_causal=bool(cfg.get("is_causal", True)),
    )

    weights = load_audio_vae_weights(model_path)
    sanitized = sanitize_audio_vae_weights(weights)
    enc_weights = {k.replace("encoder.", ""): v for k, v in sanitized.items() if k.startswith("encoder.")}
    if enc_weights:
        encoder.load_weights(list(enc_weights.items()), strict=False)
    if "per_channel_statistics._mean_of_means" in sanitized:
        encoder.per_channel_statistics._mean_of_means = sanitized["per_channel_statistics._mean_of_means"]
    if "per_channel_statistics._std_of_means" in sanitized:
        encoder.per_channel_statistics._std_of_means = sanitized["per_channel_statistics._std_of_means"]

    processor = AudioProcessor(
        sample_rate=sample_rate,
        mel_bins=mel_bins,
        mel_hop_length=mel_hop_length,
        n_fft=n_fft,
    )

    return encoder, processor


def _extract_audio_pcm(path: Path, sample_rate: int, channels: int = 2) -> Tuple[np.ndarray, int] | None:
    """Extract PCM audio using ffmpeg. Returns (waveform[channels, samples], sample_rate) or None."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    data = np.frombuffer(proc.stdout, dtype=np.int16)
    if data.size == 0:
        return None
    waveform = data.reshape(-1, channels).T.astype(np.float32) / 32768.0
    return waveform, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute latents/conditions for MLX trainer")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder of videos/images")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompts-file", type=str, default=None, help="JSON or txt mapping file->prompt")
    parser.add_argument("--caption", action="store_true", help="Auto-caption if prompt missing")
    parser.add_argument("--caption-model", type=str, default="mlx-community/SmolVLM-Instruct-4bit")
    parser.add_argument("--caption-backend", type=str, default="mlx_vlm", choices=["mlx_vlm", "transformers"])
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2")
    parser.add_argument("--text-encoder-repo", type=str, default=None)
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--audio-latents-dir", type=str, default=None, help="Copy precomputed audio latents from this dir")
    parser.add_argument("--reference-dir", type=str, default=None, help="Optional reference videos for video_to_video")
    parser.add_argument("--frame-cap", type=int, default=None)
    parser.add_argument("--resolution-buckets", type=str, default=None, help="Optional buckets: WxHxF;WxHxF")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = output_dir / "latents"
    cond_dir = output_dir / "conditions"
    latents_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = None
    ref_out_dir = None
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        ref_out_dir = output_dir / "reference_latents"
        ref_out_dir.mkdir(parents=True, exist_ok=True)

    if args.with_audio and args.audio_latents_dir:
        audio_src = Path(args.audio_latents_dir)
        audio_out = output_dir / "audio_latents"
        audio_out.mkdir(parents=True, exist_ok=True)
    else:
        audio_src = None

    prompts_map = _load_prompts(args.prompts_file)
    captioner = None
    if args.caption:
        captioner = get_captioner(args.caption_backend, args.caption_model)

    model_path = get_model_path(args.model_repo)
    encoder = load_vae_encoder(str(model_path))

    audio_encoder = None
    audio_processor = None
    if args.with_audio:
        audio_encoder, audio_processor = _build_audio_encoder(model_path)

    text_encoder_repo = args.text_encoder_repo or str(model_path)
    text_encoder_path = get_model_path(text_encoder_repo) if args.text_encoder_repo else model_path
    text_encoder = LTX2TextEncoder()
    text_encoder.load(model_path=Path(model_path), text_encoder_path=str(text_encoder_path))

    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".png", ".jpg", ".jpeg"}])
    if not files:
        raise RuntimeError(f"No media files found in {input_dir}")

    buckets = None
    if args.resolution_buckets:
        buckets = []
        for item in args.resolution_buckets.split(";"):
            if not item.strip():
                continue
            w, h, f = item.split("x")
            buckets.append((int(f), int(h), int(w)))

    for path in files:
        name = path.stem
        prompt = prompts_map.get(path.name) or prompts_map.get(name)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image
            frame = np.array(Image.open(path).convert("RGB"))
            frames = np.expand_dims(frame, axis=0)
            fps = 24.0
        else:
            frames, fps = read_video(str(path), max_frames=args.frame_cap)
        frames = np.array(frames)
        # bucket to nearest requested resolution (if provided)
        if buckets:
            target_f, target_h, target_w = _select_bucket(frames, buckets)
            if args.debug:
                print(
                    f"[precompute] bucket={target_w}x{target_h}x{target_f} for {path.name} "
                    f"(src={frames.shape[2]}x{frames.shape[1]}x{frames.shape[0]})"
                )
            frames = _match_frame_count(frames, target_f)
            frames = _resize_and_crop(frames, target_h, target_w)
        else:
            frames = _ensure_frames(frames)
        if prompt is None:
            if captioner is None:
                raise RuntimeError(f"Missing prompt for {path} and captioning disabled")
            prompt = captioner.caption(frames[0])

        if args.debug:
            print(f"[precompute] {path.name}: frames={frames.shape}, fps={fps}, prompt_len={len(prompt)}")

        latents = encode_video_latents(frames, encoder)
        latents_payload = {
            "latents": latents.astype(np.float32),
            "num_frames": np.array([latents.shape[2]], dtype=np.int32),
            "height": np.array([latents.shape[3]], dtype=np.int32),
            "width": np.array([latents.shape[4]], dtype=np.int32),
            "fps": np.array([fps], dtype=np.float32),
        }
        save_file(latents_payload, str(latents_dir / f"{name}.safetensors"))

        video_embeds, audio_embeds = text_encoder.encode(prompt, max_length=1024, return_audio_embeddings=True)
        # Cast to float32 for safetensors compatibility (numpy has limited bfloat16 support).
        cond_payload = {
            "video_prompt_embeds": np.array(video_embeds.astype(mx.float32)),
            "audio_prompt_embeds": np.array(audio_embeds.astype(mx.float32)),
            "prompt_attention_mask": np.ones((video_embeds.shape[1],), dtype=bool),
        }
        save_file(cond_payload, str(cond_dir / f"{name}.safetensors"))

        if ref_dir and ref_out_dir:
            ref_path = ref_dir / path.name
            if ref_path.exists():
                if ref_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    from PIL import Image
                    frame = np.array(Image.open(ref_path).convert("RGB"))
                    ref_frames = np.expand_dims(frame, axis=0)
                else:
                    ref_frames, _ = read_video(str(ref_path), max_frames=args.frame_cap)
                ref_frames = _ensure_frames(np.array(ref_frames))
                ref_latents = encode_video_latents(ref_frames, encoder)
                ref_payload = {
                    "latents": ref_latents.astype(np.float32),
                    "num_frames": np.array([ref_latents.shape[2]], dtype=np.int32),
                    "height": np.array([ref_latents.shape[3]], dtype=np.int32),
                    "width": np.array([ref_latents.shape[4]], dtype=np.int32),
                    "fps": np.array([fps], dtype=np.float32),
                }
                save_file(ref_payload, str(ref_out_dir / f"{name}.safetensors"))

        if audio_src is not None:
            src = audio_src / f"{name}.safetensors"
            if src.exists():
                (output_dir / "audio_latents").mkdir(parents=True, exist_ok=True)
                (output_dir / "audio_latents" / f"{name}.safetensors").write_bytes(src.read_bytes())
            else:
                print(f"[precompute] Missing audio latents for {name}, skipping.")
        elif args.with_audio and audio_encoder is not None and audio_processor is not None:
            audio_out = output_dir / "audio_latents"
            audio_out.mkdir(parents=True, exist_ok=True)
            extracted = _extract_audio_pcm(path, audio_processor.sample_rate)
            if extracted is None:
                print(f"[precompute] No audio track for {path.name}, skipping audio latents.")
            else:
                waveform, sr = extracted
                mel = audio_processor.waveform_to_mel(waveform, sr)
                mel_mx = mx.array(mel)
                audio_latents = audio_encoder(mel_mx)
                audio_latents_np = np.array(audio_latents)
                if args.debug:
                    print(
                        f"[precompute] audio {path.name}: waveform={waveform.shape}, mel={mel.shape}, "
                        f"latents={audio_latents_np.shape}"
                    )
                audio_payload = {
                    "latents": audio_latents_np[0],
                    "num_time_steps": np.array([audio_latents_np.shape[2]], dtype=np.int32),
                    "frequency_bins": np.array([audio_latents_np.shape[3]], dtype=np.int32),
                    "duration": np.array([waveform.shape[1] / float(sr)], dtype=np.float32),
                }
                save_file(audio_payload, str(audio_out / f"{name}.safetensors"))


if __name__ == "__main__":
    main()
