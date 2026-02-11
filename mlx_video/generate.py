"""Unified video and audio-video generation pipeline for LTX-2.

Supports both distilled (two-stage with upsampling) and dev (single-stage with CFG) pipelines.
"""

import argparse
import os
import re
import math
import time
import json
import shutil
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel

# Rich console for styled output
console = Console()

def _format_eta(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS (matches backend ETA parsing expectations)."""
    try:
        seconds_i = int(round(float(seconds)))
    except Exception:
        seconds_i = 0
    seconds_i = max(0, seconds_i)
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _ui_event(payload: dict) -> None:
    """Emit a single-line JSON progress event for UIs (Pinokio / web frontends).

    This stays off by default and is enabled by setting `MLX_VIDEO_UI_JSON=1` in
    the environment. The UI can safely parse lines starting with
    `MLX_VIDEO_UI_EVENT `.
    """

    if os.environ.get("MLX_VIDEO_UI_JSON") != "1":
        return
    try:
        print(
            "MLX_VIDEO_UI_EVENT "
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            flush=True,
        )
    except Exception:
        # Never let UI logging break generation.
        pass


class _PhaseTimer:
    """Lightweight phase timer for profiling end-to-end generation."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.times_s: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times_s[name] = self.times_s.get(name, 0.0) + (time.perf_counter() - t0)

    def render(self, elapsed_s: float) -> str:
        if not self.times_s:
            return ""
        total = sum(self.times_s.values())
        denom = total if total > 0 else elapsed_s if elapsed_s > 0 else 1.0
        rows = sorted(self.times_s.items(), key=lambda kv: kv[1], reverse=True)
        lines = []
        for name, dt in rows:
            pct = 100.0 * (dt / denom)
            lines.append(f"{name:>18}: {dt:6.2f}s ({pct:5.1f}%)")
        lines.append(f"{'total(phases)':>18}: {total:6.2f}s")
        lines.append(f"{'elapsed(wall)':>18}: {elapsed_s:6.2f}s")
        return "\n".join(lines)


def _debug_enabled() -> bool:
    # Keep backward compatibility with older local wrappers that used MLX_VIDEO_DEBUG.
    return os.environ.get("LTX_DEBUG") == "1" or os.environ.get("MLX_VIDEO_DEBUG") == "1"


def _debug_log(message: str) -> None:
    if _debug_enabled():
        console.print(f"[dim][debug] {message}[/]")


def _debug_stats(name: str, tensor: Optional[mx.array]) -> None:
    if not _debug_enabled() or tensor is None:
        return
    try:
        t = tensor.astype(mx.float32)
        mn = mx.min(t)
        mxv = mx.max(t)
        mean = mx.mean(t)
        std = mx.std(t)
        mx.eval(mn, mxv, mean, std)
        console.print(
            "[dim][debug] "
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"min={float(mn.item()):.6f} max={float(mxv.item()):.6f} "
            f"mean={float(mean.item()):.6f} std={float(std.item()):.6f}[/]"
        )
    except Exception as exc:
        console.print(f"[dim][debug] {name}: <stats error> {exc}[/]")


def _debug_weights_summary(label: str, weights_path: Optional[Path]) -> None:
    if not _debug_enabled():
        return
    try:
        if weights_path is None:
            _debug_log(f"{label}: <missing path>")
            return
        weights_path = Path(weights_path)
        if not weights_path.exists():
            _debug_log(f"{label}: missing {weights_path}")
            return
        weights = mx.load(str(weights_path))
        total = len(weights)
        scales = sum(1 for k in weights if k.endswith(".scales"))
        biases = sum(1 for k in weights if k.endswith(".biases"))
        dtypes = {}
        for v in weights.values():
            key = str(v.dtype)
            dtypes[key] = dtypes.get(key, 0) + 1
        _debug_log(f"{label}: {weights_path} keys={total} scales={scales} biases={biases} dtypes={dtypes}")
    except Exception as exc:
        _debug_log(f"{label}: <failed to load> {exc}")


def _looks_like_metal_oom(exc: BaseException) -> bool:
    """Best-effort detection of Metal OOM/resource exhaustion errors."""
    msg = str(exc)
    needles = (
        "out of memory",
        "Out of memory",
        "OOM",
        "failed to allocate",
        "kIOGPU",
        "Command buffer execution failed",
        "Invalid Resource",
        "MTLCommandBufferError",
        "[METAL]",
    )
    return any(n in msg for n in needles)


def _subsample_sigmas_farthest(sigmas: list[float], steps: int) -> list[float]:
    """Subsample a fixed sigma schedule to `steps` denoise steps.

    The distilled schedules include several near-identical high-sigma entries.
    When reducing step counts, selecting indices uniformly can over-sample that
    region and drop useful mid/low-sigma points. This routine picks a diverse
    subset in log-sigma space (farthest-point sampling) to preserve quality at
    lower step counts.

    Always includes the first and last sigmas.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    max_steps = len(sigmas) - 1
    if steps >= max_steps:
        return sigmas
    if steps == 1:
        return [sigmas[0], sigmas[-1]]

    # Avoid log(0) by selecting among non-zero sigmas and always appending the
    # final 0.0 endpoint.
    eps = 1e-6
    pool = sigmas[:-1]
    xs = [math.log(max(s, eps)) for s in pool]
    chosen = {0, len(pool) - 1}
    while len(chosen) < steps:
        best_i = None
        best_score = -1.0
        for i in range(len(pool)):
            if i in chosen:
                continue
            score = min(abs(xs[i] - xs[j]) for j in chosen)
            if score > best_score:
                best_score = score
                best_i = i
        assert best_i is not None
        chosen.add(best_i)

    idxs = sorted(chosen)
    return [sigmas[i] for i in idxs] + [sigmas[-1]]


def _subsample_sigmas_uniform(sigmas: list[float], steps: int) -> list[float]:
    """Uniformly subsample a fixed sigma schedule to `steps` denoise steps.

    Always includes the first and last sigmas.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    max_steps = len(sigmas) - 1
    if steps >= max_steps:
        return sigmas
    if steps == 1:
        return [sigmas[0], sigmas[-1]]

    pool = sigmas[:-1]
    last = len(pool) - 1
    idxs: list[int] = [0]
    for i in range(1, steps - 1):
        idxs.append(int(round(i * last / (steps - 1))))
    idxs.append(last)

    # Round+int can create duplicates for small schedules; fill missing indices.
    uniq = sorted(set(idxs))
    if len(uniq) < steps:
        for i in range(last + 1):
            if i in uniq:
                continue
            uniq.append(i)
            if len(uniq) == steps:
                break
        uniq = sorted(uniq)

    return [pool[i] for i in uniq] + [sigmas[-1]]


def _subsample_sigmas(sigmas: list[float], steps: int, method: str) -> list[float]:
    if method == "uniform":
        return _subsample_sigmas_uniform(sigmas, steps)
    if method == "farthest":
        return _subsample_sigmas_farthest(sigmas, steps)
    raise ValueError(f"Unknown sigma subsample method: {method}")


def _subsample_refinement_sigmas(sigmas: list[float], steps: int, method: str) -> list[float]:
    """Subsample sigma schedules for refinement passes.

    For the stage-2 refinement schedule, a single denoise step is most useful at
    low sigma (i.e., close to clean) because the model has only one chance to
    add details. Starting from very high sigma injects a lot of noise and can
    wash out the stage-1 structure when `steps=1`.
    """
    if steps == 1 and method == "farthest" and len(sigmas) >= 3:
        # Pick the last non-zero sigma as the starting point.
        return [sigmas[-2], sigmas[-1]]
    return _subsample_sigmas(sigmas, steps, method)

from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality

from mlx_video.utils import (
    to_denoised,
    load_image,
    load_video,
    prepare_image_for_encoding,
    prepare_video_for_encoding,
    get_model_path,
)
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.models.ltx.video_vae.tiling import TilingConfig
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents
from mlx_video.conditioning import VideoConditionByLatentIndex, VideoConditionByKeyframeIndex, apply_conditioning
from mlx_video.conditioning.latent import LatentState, apply_denoise_mask


class PipelineType(Enum):
    """Pipeline type selector."""
    DISTILLED = "distilled"  # Two-stage with upsampling, fixed sigmas, no CFG
    DEV = "dev"              # Single-stage, dynamic sigmas, CFG
    KEYFRAME = "keyframe"    # Two-stage, guiding keyframes (KFI)
    IC_LORA = "ic_lora"      # Two-stage, IC-LoRA with video conditioning


def _bytes_to_gb(value: float) -> float:
    return value / (1024 ** 3)


def _get_memory_stats() -> Tuple[float, float, float]:
    active = mx.get_active_memory()
    peak = mx.get_peak_memory()
    cache = 0.0
    if hasattr(mx, "get_cache_memory"):
        try:
            cache = mx.get_cache_memory()
        except Exception:
            cache = 0.0
    elif hasattr(mx, "metal"):
        try:
            cache = mx.metal.get_cache_memory()
        except Exception:
            cache = 0.0
    return active, cache, peak


def _log_memory(stage: str, enabled: bool) -> None:
    if not enabled:
        return
    active, cache, peak = _get_memory_stats()
    console.print(
        f"[dim]Memory ({stage}): active={_bytes_to_gb(active):.2f}GB, "
        f"cache={_bytes_to_gb(cache):.2f}GB, peak={_bytes_to_gb(peak):.2f}GB[/]"
    )


# Distilled model sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

# Dev model scheduling constants
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR  # 25

# Default negative prompt for CFG (dev pipeline)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)



def _slugify_filename(text: str, max_len: int = 80) -> str:
    """Create a filesystem-safe slug from a string."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "video"
    return text[:max_len].strip("-")


def cfg_delta(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
    """Compute CFG delta for classifier-free guidance.

    Args:
        cond: Conditional prediction
        uncond: Unconditional prediction
        scale: CFG guidance scale

    Returns:
        Delta to add to unconditional for CFG: (scale - 1) * (cond - uncond)
    """
    return (scale - 1.0) * (cond - uncond)


def _start_metal_capture(path: Path) -> bool:
    """Start a Metal GPU capture if available.

    Returns True when a capture started successfully.
    """
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        console.print("[yellow]⚠️  Metal capture requested, but Metal backend is unavailable.[/]")
        return False
    if path.exists():
        raise FileExistsError(f"Capture path already exists: {path}")
    mx.metal.start_capture(str(path))
    return True


def ltx2_scheduler(
    steps: int,
    num_tokens: Optional[int] = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> mx.array:
    """LTX-2 scheduler for sigma generation (dev model).

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.

    Args:
        steps: Number of inference steps
        num_tokens: Number of latent tokens (F*H*W). If None, uses MAX_SHIFT_ANCHOR
        max_shift: Maximum shift factor
        base_shift: Base shift factor
        stretch: Whether to stretch sigmas to terminal value
        terminal: Terminal sigma value for stretching

    Returns:
        Array of sigma values of shape (steps + 1,)
    """
    if num_tokens is None:
        tokens = MAX_SHIFT_ANCHOR
    else:
        tokens = min(num_tokens, MAX_SHIFT_ANCHOR)
    sigmas = np.linspace(1.0, 0.0, steps + 1)

    # Compute shift based on token count
    x1 = BASE_SHIFT_ANCHOR
    x2 = MAX_SHIFT_ANCHOR
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b

    # Apply shift transformation
    power = 1
    # Avoid divide-by-zero warnings by applying the transform only to non-zero sigmas
    transformed = np.zeros_like(sigmas)
    non_zero = sigmas != 0
    if np.any(non_zero):
        nz = sigmas[non_zero]
        transformed[non_zero] = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / nz - 1) ** power)
    sigmas = transformed

    # Stretch sigmas to terminal value
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        if np.isfinite(scale_factor) and scale_factor != 0:
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

    return mx.array(sigmas, dtype=mx.float32)


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for RoPE in pixel space.

    Args:
        batch_size: Batch size
        num_frames: Number of frames (latent)
        height: Height (latent)
        width: Width (latent)
        temporal_scale: VAE temporal scale factor (default 8)
        spatial_scale: VAE spatial scale factor (default 32)
        fps: Frames per second (default 24.0)
        causal_fix: Apply causal fix for first frame (default True)

    Returns:
        Position grid of shape (B, 3, num_patches, 2) in pixel space
        where dim 2 is [start, end) bounds for each patch
    """
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None
        )

    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    return mx.array(pixel_coords, dtype=mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Create temporal position grid for audio RoPE."""
    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    start_times = get_audio_latent_time_in_sec(0, audio_frames)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    return mx.array(positions, dtype=mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


# =============================================================================
# Distilled Pipeline Denoising (no CFG, fixed sigmas)
# =============================================================================

def denoise_distilled(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    audio_latents: Optional[mx.array] = None,
    audio_positions: Optional[mx.array] = None,
    audio_embeddings: Optional[mx.array] = None,
    eval_interval: int = 1,
    compile_step: bool = False,
    compile_shapeless: bool = False,
    fp32_euler: bool = True,
    ui_phase: str = "denoise",
) -> tuple[mx.array, Optional[mx.array]]:
    """Run denoising loop for distilled pipeline (no CFG)."""
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = latents.dtype
    enable_audio = audio_latents is not None

    if state is not None:
        latents = state.latent

    desc = "[cyan]Denoising A/V[/]" if enable_audio else "[cyan]Denoising[/]"
    num_steps = len(sigmas) - 1
    sigmas_mx = mx.array(sigmas, dtype=dtype)

    b, c, f, h, w = latents.shape
    num_tokens = f * h * w

    denoise_mask_flat = None
    if state is not None:
        denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
        denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
        denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))

    if denoise_mask_flat is None:
        video_timesteps_mask = mx.ones((b, num_tokens), dtype=dtype)
    else:
        video_timesteps_mask = denoise_mask_flat.astype(dtype)

    audio_timesteps_mask = None
    if enable_audio:
        ab, ac, at, af = audio_latents.shape
        audio_timesteps_mask = mx.ones((ab, at), dtype=dtype)

    # Precompute RoPE once (distilled path previously recomputed this inside the model per step).
    precomputed_video_rope = precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    precomputed_audio_rope = None
    if enable_audio:
        if audio_positions is None or audio_embeddings is None:
            raise ValueError("audio_positions/audio_embeddings must be provided when audio_latents is enabled")
        precomputed_audio_rope = precompute_freqs_cis(
            audio_positions,
            dim=transformer.audio_inner_dim,
            theta=transformer.positional_embedding_theta,
            max_pos=transformer.audio_positional_embedding_max_pos,
            use_middle_indices_grid=transformer.use_middle_indices_grid,
            num_attention_heads=transformer.audio_num_attention_heads,
            rope_type=transformer.rope_type,
            double_precision=transformer.config.double_precision_rope,
        )
        mx.eval(precomputed_video_rope, precomputed_audio_rope)
    else:
        mx.eval(precomputed_video_rope)

    if compile_step and enable_audio:
        def step_fn(
            latents_in: mx.array,
            audio_latents_in: mx.array,
            sigma_in: mx.array,
            sigma_next_in: mx.array,
        ) -> tuple[mx.array, mx.array]:
            b, c, f, h, w = latents_in.shape
            num_tokens = f * h * w
            latents_flat = mx.transpose(mx.reshape(latents_in, (b, c, -1)), (0, 2, 1))

            timesteps = sigma_in * video_timesteps_mask

            video_modality = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
            )

            ab, ac, at, af = audio_latents_in.shape
            audio_flat = mx.transpose(audio_latents_in, (0, 2, 1, 3))
            audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

            audio_modality = Modality(
                latent=audio_flat,
                timesteps=sigma_in * audio_timesteps_mask,
                positions=audio_positions,
                context=audio_embeddings,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
            )

            velocity, audio_velocity = transformer(video=video_modality, audio=audio_modality)
            velocity = mx.reshape(mx.transpose(velocity, (0, 2, 1)), (b, c, f, h, w))
            denoised = to_denoised(latents_in, velocity, sigma_in)

            audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
            audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
            audio_denoised = to_denoised(audio_latents_in, audio_velocity, sigma_in)

            if state is not None:
                denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

            if fp32_euler:
                latents_f32 = latents_in.astype(mx.float32)
                denoised_f32 = denoised.astype(mx.float32)
                sigma_next_f32 = sigma_next_in.astype(mx.float32)
                sigma_f32 = sigma_in.astype(mx.float32)
                latents_out = (denoised_f32 + sigma_next_f32 * (latents_f32 - denoised_f32) / sigma_f32).astype(dtype)
            else:
                latents_out = (denoised + sigma_next_in * (latents_in - denoised) / sigma_in).astype(dtype)

            if fp32_euler:
                audio_latents_f32 = audio_latents_in.astype(mx.float32)
                audio_denoised_f32 = audio_denoised.astype(mx.float32)
                audio_latents_out = (
                    audio_denoised_f32 + sigma_next_f32 * (audio_latents_f32 - audio_denoised_f32) / sigma_f32
                ).astype(dtype)
            else:
                audio_latents_out = (
                    audio_denoised + sigma_next_in * (audio_latents_in - audio_denoised) / sigma_in
                ).astype(dtype)

            return latents_out, audio_latents_out

        step_fn = mx.compile(step_fn, shapeless=compile_shapeless)
    elif compile_step and not enable_audio:
        def step_fn(
            latents_in: mx.array,
            sigma_in: mx.array,
            sigma_next_in: mx.array,
        ) -> mx.array:
            b, c, f, h, w = latents_in.shape
            num_tokens = f * h * w
            latents_flat = mx.transpose(mx.reshape(latents_in, (b, c, -1)), (0, 2, 1))
            timesteps = sigma_in * video_timesteps_mask

            video_modality = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
            )

            velocity, _ = transformer(video=video_modality, audio=None)
            velocity = mx.reshape(mx.transpose(velocity, (0, 2, 1)), (b, c, f, h, w))
            denoised = to_denoised(latents_in, velocity, sigma_in)

            if state is not None:
                denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

            if fp32_euler:
                latents_f32 = latents_in.astype(mx.float32)
                denoised_f32 = denoised.astype(mx.float32)
                sigma_next_f32 = sigma_next_in.astype(mx.float32)
                sigma_f32 = sigma_in.astype(mx.float32)
                latents_out = (denoised_f32 + sigma_next_f32 * (latents_f32 - denoised_f32) / sigma_f32).astype(dtype)
            else:
                latents_out = (denoised + sigma_next_in * (latents_in - denoised) / sigma_in).astype(dtype)
            return latents_out

        step_fn = mx.compile(step_fn, shapeless=compile_shapeless)
    else:
        step_fn = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(desc, total=num_steps)
        progress_echo = os.environ.get("MLX_VIDEO_PROGRESS_ECHO") == "1"
        try:
            progress_echo_every = int(
                os.environ.get(
                    "MLX_VIDEO_DENOISE_ECHO_EVERY",
                    os.environ.get("MLX_VIDEO_PROGRESS_ECHO_EVERY", "12"),
                )
            )
        except Exception:
            progress_echo_every = 12
        last_echo_i = -1
        t0 = time.perf_counter()
        # Strip Rich markup so stdout scraping stays stable.
        desc_plain = re.sub(r"\[[^\]]+\]", "", desc).strip() or "Denoising"

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_mx = sigmas_mx[i]
            sigma_next_mx = sigmas_mx[i + 1]

            if step_fn is not None:
                if enable_audio:
                    latents, audio_latents = step_fn(latents, audio_latents, sigma_mx, sigma_next_mx)
                else:
                    latents = step_fn(latents, sigma_mx, sigma_next_mx)
            else:
                latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
                timesteps = sigma_mx * video_timesteps_mask

                video_modality = Modality(
                    latent=latents_flat,
                    timesteps=timesteps,
                    positions=positions,
                    context=text_embeddings,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_video_rope,
                )

                audio_modality = None
                if enable_audio:
                    audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
                    audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

                    audio_modality = Modality(
                        latent=audio_flat,
                        timesteps=sigma_mx * audio_timesteps_mask,
                        positions=audio_positions,
                        context=audio_embeddings,
                        context_mask=None,
                        enabled=True,
                        positional_embeddings=precomputed_audio_rope,
                    )

                velocity, audio_velocity = transformer(video=video_modality, audio=audio_modality)

                velocity = mx.reshape(mx.transpose(velocity, (0, 2, 1)), (b, c, f, h, w))
                denoised = to_denoised(latents, velocity, sigma_mx)

                audio_denoised = None
                if enable_audio and audio_velocity is not None:
                    ab, ac, at, af = audio_latents.shape
                    audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
                    audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
                    audio_denoised = to_denoised(audio_latents, audio_velocity, sigma_mx)

                if state is not None:
                    denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

                if sigma_next > 0:
                    # Compute Euler step in float32 for precision (matching PyTorch behavior)
                    latents_f32 = latents.astype(mx.float32)
                    denoised_f32 = denoised.astype(mx.float32)
                    sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                    sigma_f32 = mx.array(sigma, dtype=mx.float32)
                    latents = (denoised_f32 + sigma_next_f32 * (latents_f32 - denoised_f32) / sigma_f32).astype(dtype)
                    if enable_audio and audio_denoised is not None:
                        audio_latents_f32 = audio_latents.astype(mx.float32)
                        audio_denoised_f32 = audio_denoised.astype(mx.float32)
                        audio_latents = (audio_denoised_f32 + sigma_next_f32 * (audio_latents_f32 - audio_denoised_f32) / sigma_f32).astype(dtype)
                else:
                    latents = denoised
                    if enable_audio and audio_denoised is not None:
                        audio_latents = audio_denoised

            if (i + 1) % eval_interval == 0 or i == num_steps - 1:
                if enable_audio:
                    mx.eval(latents, audio_latents)
                else:
                    mx.eval(latents)

            progress.advance(task)
            done = i + 1
            elapsed = time.perf_counter() - t0
            eta_s = (elapsed / max(1, done)) * max(0, num_steps - done)

            if progress_echo and progress_echo_every > 0:
                if i == 0 or i == num_steps - 1 or (i - last_echo_i) >= progress_echo_every:
                    print(f"{desc_plain} {done}/{num_steps} ETA {_format_eta(eta_s)}", flush=True)
                    last_echo_i = i

            _ui_event(
                {
                    "kind": "progress",
                    "phase": ui_phase,
                    "current": done,
                    "total": num_steps,
                    "percent": 100.0 * done / max(1, num_steps),
                    "eta_seconds": float(eta_s),
                }
            )

    _debug_stats("latents_after_denoise_distilled", latents)
    if enable_audio:
        _debug_stats("audio_latents_after_denoise_distilled", audio_latents)
    return latents, audio_latents if enable_audio else None


# =============================================================================
# Audio-only Denoising (no CFG, fixed sigmas)
# =============================================================================

def denoise_audio_only(
    audio_latents: mx.array,
    audio_positions: mx.array,
    audio_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list[float],
    verbose: bool = True,
    eval_interval: int = 1,
    compile_step: bool = False,
    compile_shapeless: bool = False,
    fp32_euler: bool = True,
    ui_phase: str = "audio_denoise",
) -> mx.array:
    """Run a distilled-style denoising loop for AudioOnly transformers.

    This is used for "separate" audio generation (generate video first, then audio
    with an AudioOnly model). We intentionally do not use CFG here; for the current
    audio branch, CFG can collapse to near-silence.
    """
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = audio_latents.dtype
    num_steps = len(sigmas) - 1
    sigmas_mx = mx.array(sigmas, dtype=dtype)

    ab, ac, at, af = audio_latents.shape
    audio_timesteps_mask = mx.ones((ab, at), dtype=dtype)

    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_audio_rope)

    step_fn = None
    if compile_step:
        def _step_fn(
            audio_latents_in: mx.array,
            sigma_in: mx.array,
            sigma_next_in: mx.array,
        ) -> mx.array:
            ab, ac, at, af = audio_latents_in.shape
            audio_flat = mx.transpose(audio_latents_in, (0, 2, 1, 3))
            audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

            audio_modality = Modality(
                latent=audio_flat,
                timesteps=sigma_in * audio_timesteps_mask,
                positions=audio_positions,
                context=audio_embeddings,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
            )

            _, audio_velocity = transformer(video=None, audio=audio_modality)
            audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
            audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
            audio_denoised = to_denoised(audio_latents_in, audio_velocity, sigma_in)

            if fp32_euler:
                audio_latents_f32 = audio_latents_in.astype(mx.float32)
                audio_denoised_f32 = audio_denoised.astype(mx.float32)
                sigma_next_f32 = sigma_next_in.astype(mx.float32)
                sigma_f32 = sigma_in.astype(mx.float32)
                audio_latents_out = (
                    audio_denoised_f32 + sigma_next_f32 * (audio_latents_f32 - audio_denoised_f32) / sigma_f32
                ).astype(dtype)
            else:
                audio_latents_out = (
                    audio_denoised + sigma_next_in * (audio_latents_in - audio_denoised) / sigma_in
                ).astype(dtype)
            return audio_latents_out

        step_fn = mx.compile(_step_fn, shapeless=compile_shapeless)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("[cyan]Denoising audio[/]", total=num_steps)
        progress_echo = os.environ.get("MLX_VIDEO_PROGRESS_ECHO") == "1"
        try:
            progress_echo_every = int(
                os.environ.get(
                    "MLX_VIDEO_DENOISE_ECHO_EVERY",
                    os.environ.get("MLX_VIDEO_PROGRESS_ECHO_EVERY", "12"),
                )
            )
        except Exception:
            progress_echo_every = 12
        last_echo_i = -1
        t0 = time.perf_counter()
        desc_plain = "Denoising audio"

        for i in range(num_steps):
            sigma_mx = sigmas_mx[i]
            sigma_next_mx = sigmas_mx[i + 1]

            if step_fn is not None:
                audio_latents = step_fn(audio_latents, sigma_mx, sigma_next_mx)
            else:
                audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
                audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

                audio_modality = Modality(
                    latent=audio_flat,
                    timesteps=sigma_mx * audio_timesteps_mask,
                    positions=audio_positions,
                    context=audio_embeddings,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                )

                _, audio_velocity = transformer(video=None, audio=audio_modality)
                audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
                audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
                audio_denoised = to_denoised(audio_latents, audio_velocity, sigma_mx)

                # Euler step (float32 for precision)
                if sigmas[i + 1] > 0:
                    sigma_next_f32 = mx.array(sigmas[i + 1], dtype=mx.float32)
                    sigma_f32 = mx.array(sigmas[i], dtype=mx.float32)
                    audio_latents_f32 = audio_latents.astype(mx.float32)
                    audio_denoised_f32 = audio_denoised.astype(mx.float32)
                    audio_latents = (audio_denoised_f32 + sigma_next_f32 * (audio_latents_f32 - audio_denoised_f32) / sigma_f32).astype(dtype)
                else:
                    audio_latents = audio_denoised

            if (i + 1) % eval_interval == 0 or i == num_steps - 1:
                mx.eval(audio_latents)
            progress.advance(task)
            done = i + 1
            elapsed = time.perf_counter() - t0
            eta_s = (elapsed / max(1, done)) * max(0, num_steps - done)

            if progress_echo and progress_echo_every > 0:
                if i == 0 or i == num_steps - 1 or (i - last_echo_i) >= progress_echo_every:
                    print(f"{desc_plain} {done}/{num_steps} ETA {_format_eta(eta_s)}", flush=True)
                    last_echo_i = i

            _ui_event(
                {
                    "kind": "progress",
                    "phase": ui_phase,
                    "current": done,
                    "total": num_steps,
                    "percent": 100.0 * done / max(1, num_steps),
                    "eta_seconds": float(eta_s),
                }
            )

    _debug_stats("audio_latents_after_denoise_audio_only", audio_latents)
    return audio_latents


# =============================================================================
# Dev Pipeline Denoising (with CFG, dynamic sigmas)
# =============================================================================

def denoise_dev(
    latents: mx.array,
    positions: mx.array,
    text_embeddings_pos: mx.array,
    text_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    eval_interval: int = 1,
    compile_step: bool = False,
    compile_shapeless: bool = False,
    cfg_batch: bool = False,
    ui_phase: str = "denoise",
) -> mx.array:
    """Run denoising loop for dev pipeline with CFG."""
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = latents.dtype
    if state is not None:
        latents = state.latent

    sigmas_list = sigmas.tolist()
    sigmas_mx = sigmas.astype(dtype)
    use_cfg = cfg_scale != 1.0
    cfg_batch = cfg_batch and use_cfg
    num_steps = len(sigmas_list) - 1

    b, c, f, h, w = latents.shape
    num_tokens = f * h * w

    # Precompute once: constant CFG contexts (pos/neg) for cfg_batch path.
    context_cat = None
    if cfg_batch:
        context_cat = mx.concatenate([text_embeddings_pos, text_embeddings_neg], axis=0)

    denoise_mask_flat = None
    if state is not None:
        denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
        denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
        denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))

    if denoise_mask_flat is None:
        video_timesteps_mask = mx.ones((b, num_tokens), dtype=dtype)
    else:
        video_timesteps_mask = denoise_mask_flat.astype(dtype)

    if compile_step:
        def step_fn(
            latents_in: mx.array,
            sigma_in: mx.array,
            sigma_next_in: mx.array,
        ) -> mx.array:
            b, c, f, h, w = latents_in.shape
            num_tokens = f * h * w
            latents_flat = mx.transpose(mx.reshape(latents_in, (b, c, -1)), (0, 2, 1))
            timesteps = sigma_in * video_timesteps_mask

            if cfg_batch:
                # Avoid per-step materialization copies for CFG batching.
                latents_cat = mx.broadcast_to(mx.expand_dims(latents_flat, axis=0), (2,) + latents_flat.shape)
                latents_cat = mx.reshape(latents_cat, (2 * b,) + latents_flat.shape[1:])
                timesteps_cat = mx.broadcast_to(mx.expand_dims(timesteps, axis=0), (2,) + timesteps.shape)
                timesteps_cat = mx.reshape(timesteps_cat, (2 * b,) + timesteps.shape[1:])
                video_modality = Modality(
                    latent=latents_cat,
                    timesteps=timesteps_cat,
                    positions=positions_cfg,
                    context=context_cat,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_rope_cfg,
                )
                velocity_cat, _ = transformer(video=video_modality, audio=None)
                velocity_pos, velocity_neg = mx.split(velocity_cat, 2, axis=0)
                velocity_flat = velocity_pos + (cfg_scale - 1.0) * (velocity_pos - velocity_neg)
            else:
                video_modality_pos = Modality(
                    latent=latents_flat,
                    timesteps=timesteps,
                    positions=positions,
                    context=text_embeddings_pos,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                )
                velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

                if use_cfg:
                    video_modality_neg = Modality(
                        latent=latents_flat,
                        timesteps=timesteps,
                        positions=positions,
                        context=text_embeddings_neg,
                        context_mask=None,
                        enabled=True,
                        positional_embeddings=precomputed_rope,
                    )
                    velocity_neg, _ = transformer(video=video_modality_neg, audio=None)
                    velocity_flat = velocity_pos + (cfg_scale - 1.0) * (velocity_pos - velocity_neg)
                else:
                    velocity_flat = velocity_pos

            velocity = mx.reshape(mx.transpose(velocity_flat, (0, 2, 1)), (b, c, f, h, w))
            denoised = to_denoised(latents_in, velocity, sigma_in)

            if state is not None:
                denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

            latents_f32 = latents_in.astype(mx.float32)
            denoised_f32 = denoised.astype(mx.float32)
            sigma_next_f32 = sigma_next_in.astype(mx.float32)
            sigma_f32 = sigma_in.astype(mx.float32)
            latents_out = (denoised_f32 + sigma_next_f32 * (latents_f32 - denoised_f32) / sigma_f32).astype(dtype)
            return latents_out

        step_fn = mx.compile(step_fn, shapeless=compile_shapeless)
    else:
        step_fn = None

    # Precompute RoPE once
    precomputed_rope = precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_rope)

    positions_cfg = positions
    precomputed_rope_cfg = precomputed_rope
    if cfg_batch:
        positions_cfg = mx.broadcast_to(positions, (b * 2,) + positions.shape[1:])
        precomputed_rope_cfg = (
            mx.broadcast_to(precomputed_rope[0], (b * 2,) + precomputed_rope[0].shape[1:]),
            mx.broadcast_to(precomputed_rope[1], (b * 2,) + precomputed_rope[1].shape[1:]),
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("[cyan]Denoising (CFG)[/]", total=num_steps)
        progress_echo = os.environ.get("MLX_VIDEO_PROGRESS_ECHO") == "1"
        try:
            progress_echo_every = int(
                os.environ.get(
                    "MLX_VIDEO_DENOISE_ECHO_EVERY",
                    os.environ.get("MLX_VIDEO_PROGRESS_ECHO_EVERY", "12"),
                )
            )
        except Exception:
            progress_echo_every = 12
        last_echo_i = -1
        t0 = time.perf_counter()
        desc_plain = "Denoising"

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]
            sigma_mx = sigmas_mx[i]
            sigma_next_mx = sigmas_mx[i + 1]

            if step_fn is not None:
                latents = step_fn(latents, sigma_mx, sigma_next_mx)
            else:
                latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
                timesteps = sigma_mx * video_timesteps_mask

                if cfg_batch:
                    latents_cat = mx.broadcast_to(mx.expand_dims(latents_flat, axis=0), (2,) + latents_flat.shape)
                    latents_cat = mx.reshape(latents_cat, (2 * b,) + latents_flat.shape[1:])
                    timesteps_cat = mx.broadcast_to(mx.expand_dims(timesteps, axis=0), (2,) + timesteps.shape)
                    timesteps_cat = mx.reshape(timesteps_cat, (2 * b,) + timesteps.shape[1:])
                    video_modality = Modality(
                        latent=latents_cat,
                        timesteps=timesteps_cat,
                        positions=positions_cfg,
                        context=context_cat,
                        context_mask=None,
                        enabled=True,
                        positional_embeddings=precomputed_rope_cfg,
                    )
                    velocity_cat, _ = transformer(video=video_modality, audio=None)
                    velocity_pos, velocity_neg = mx.split(velocity_cat, 2, axis=0)
                    velocity_flat = velocity_pos + (cfg_scale - 1.0) * (velocity_pos - velocity_neg)
                else:
                    # Positive conditioning pass
                    video_modality_pos = Modality(
                        latent=latents_flat,
                        timesteps=timesteps,
                        positions=positions,
                        context=text_embeddings_pos,
                        context_mask=None,
                        enabled=True,
                        positional_embeddings=precomputed_rope,
                    )
                    velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

                    if use_cfg:
                        # Negative conditioning pass
                        video_modality_neg = Modality(
                            latent=latents_flat,
                            timesteps=timesteps,
                            positions=positions,
                            context=text_embeddings_neg,
                            context_mask=None,
                            enabled=True,
                            positional_embeddings=precomputed_rope,
                        )
                        velocity_neg, _ = transformer(video=video_modality_neg, audio=None)

                        # Apply CFG
                        velocity_flat = velocity_pos + (cfg_scale - 1.0) * (velocity_pos - velocity_neg)
                    else:
                        velocity_flat = velocity_pos

                velocity = mx.reshape(mx.transpose(velocity_flat, (0, 2, 1)), (b, c, f, h, w))
                denoised = to_denoised(latents, velocity, sigma_mx)

                if state is not None:
                    denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

                if sigma_next > 0:
                    # Compute Euler step in float32 for precision (matching PyTorch behavior)
                    latents_f32 = latents.astype(mx.float32)
                    denoised_f32 = denoised.astype(mx.float32)
                    sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                    sigma_f32 = mx.array(sigma, dtype=mx.float32)
                    latents = (denoised_f32 + sigma_next_f32 * (latents_f32 - denoised_f32) / sigma_f32).astype(dtype)
                else:
                    latents = denoised

            if (i + 1) % eval_interval == 0 or i == num_steps - 1:
                mx.eval(latents)
            progress.advance(task)
            done = i + 1
            elapsed = time.perf_counter() - t0
            eta_s = (elapsed / max(1, done)) * max(0, num_steps - done)

            if progress_echo and progress_echo_every > 0:
                if i == 0 or i == num_steps - 1 or (i - last_echo_i) >= progress_echo_every:
                    print(f"{desc_plain} {done}/{num_steps} ETA {_format_eta(eta_s)}", flush=True)
                    last_echo_i = i

            _ui_event(
                {
                    "kind": "progress",
                    "phase": ui_phase,
                    "current": done,
                    "total": num_steps,
                    "percent": 100.0 * done / max(1, num_steps),
                    "eta_seconds": float(eta_s),
                }
            )

    _debug_stats("latents_after_denoise_dev", latents)
    return latents


def denoise_dev_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    eval_interval: int = 1,
    compile_step: bool = False,
    compile_shapeless: bool = False,
    cfg_batch: bool = False,
    ui_phase: str = "denoise",
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for dev pipeline with CFG and audio."""
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    sigmas_list = sigmas.tolist()
    sigmas_mx = sigmas.astype(dtype)
    use_cfg = cfg_scale != 1.0
    cfg_batch = cfg_batch and use_cfg
    num_steps = len(sigmas_list) - 1

    b, c, f, h, w = video_latents.shape
    num_video_tokens = f * h * w

    denoise_mask_flat = None
    if video_state is not None:
        denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
        denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
        denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))

    if denoise_mask_flat is None:
        video_timesteps_mask = mx.ones((b, num_video_tokens), dtype=dtype)
    else:
        video_timesteps_mask = denoise_mask_flat.astype(dtype)

    ab, ac, at, af = audio_latents.shape
    audio_timesteps_mask = mx.ones((ab, at), dtype=dtype)

    # Precompute once: constant CFG contexts (pos/neg) for cfg_batch path.
    video_context_cat = None
    audio_context_cat = None
    if cfg_batch:
        video_context_cat = mx.concatenate([video_embeddings_pos, video_embeddings_neg], axis=0)
        audio_context_cat = mx.concatenate([audio_embeddings_pos, audio_embeddings_neg], axis=0)

    # Precompute video RoPE
    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )

    # Precompute audio RoPE
    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    video_positions_cfg = video_positions
    audio_positions_cfg = audio_positions
    precomputed_video_rope_cfg = precomputed_video_rope
    precomputed_audio_rope_cfg = precomputed_audio_rope
    if cfg_batch:
        video_positions_cfg = mx.broadcast_to(video_positions, (b * 2,) + video_positions.shape[1:])
        audio_positions_cfg = mx.broadcast_to(audio_positions, (ab * 2,) + audio_positions.shape[1:])
        precomputed_video_rope_cfg = (
            mx.broadcast_to(precomputed_video_rope[0], (b * 2,) + precomputed_video_rope[0].shape[1:]),
            mx.broadcast_to(precomputed_video_rope[1], (b * 2,) + precomputed_video_rope[1].shape[1:]),
        )
        precomputed_audio_rope_cfg = (
            mx.broadcast_to(precomputed_audio_rope[0], (ab * 2,) + precomputed_audio_rope[0].shape[1:]),
            mx.broadcast_to(precomputed_audio_rope[1], (ab * 2,) + precomputed_audio_rope[1].shape[1:]),
        )

    if compile_step:
        def step_fn(
            video_latents_in: mx.array,
            audio_latents_in: mx.array,
            sigma_in: mx.array,
            sigma_next_in: mx.array,
        ) -> tuple[mx.array, mx.array]:
            b, c, f, h, w = video_latents_in.shape
            num_video_tokens = f * h * w
            video_flat = mx.transpose(mx.reshape(video_latents_in, (b, c, -1)), (0, 2, 1))

            ab, ac, at, af = audio_latents_in.shape
            audio_flat = mx.transpose(audio_latents_in, (0, 2, 1, 3))
            audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

            video_timesteps = sigma_in * video_timesteps_mask
            audio_timesteps = sigma_in * audio_timesteps_mask

            if cfg_batch:
                # Avoid per-step materialization copies for CFG batching.
                video_latents_cat = mx.broadcast_to(mx.expand_dims(video_flat, axis=0), (2,) + video_flat.shape)
                video_latents_cat = mx.reshape(video_latents_cat, (2 * b,) + video_flat.shape[1:])
                audio_latents_cat = mx.broadcast_to(mx.expand_dims(audio_flat, axis=0), (2,) + audio_flat.shape)
                audio_latents_cat = mx.reshape(audio_latents_cat, (2 * ab,) + audio_flat.shape[1:])
                video_timesteps_cat = mx.broadcast_to(mx.expand_dims(video_timesteps, axis=0), (2,) + video_timesteps.shape)
                video_timesteps_cat = mx.reshape(video_timesteps_cat, (2 * b,) + video_timesteps.shape[1:])
                audio_timesteps_cat = mx.broadcast_to(mx.expand_dims(audio_timesteps, axis=0), (2,) + audio_timesteps.shape)
                audio_timesteps_cat = mx.reshape(audio_timesteps_cat, (2 * ab,) + audio_timesteps.shape[1:])

                video_modality = Modality(
                    latent=video_latents_cat, timesteps=video_timesteps_cat, positions=video_positions_cfg,
                    context=video_context_cat, context_mask=None, enabled=True,
                    positional_embeddings=precomputed_video_rope_cfg,
                )
                audio_modality = Modality(
                    latent=audio_latents_cat, timesteps=audio_timesteps_cat, positions=audio_positions_cfg,
                    context=audio_context_cat, context_mask=None, enabled=True,
                    positional_embeddings=precomputed_audio_rope_cfg,
                )
                video_vel_cat, audio_vel_cat = transformer(video=video_modality, audio=audio_modality)
                video_vel_pos, video_vel_neg = mx.split(video_vel_cat, 2, axis=0)
                audio_vel_pos, audio_vel_neg = mx.split(audio_vel_cat, 2, axis=0)
                video_velocity_flat = video_vel_pos + (cfg_scale - 1.0) * (video_vel_pos - video_vel_neg)
                audio_velocity_flat = audio_vel_pos + (cfg_scale - 1.0) * (audio_vel_pos - audio_vel_neg)
            else:
                video_modality_pos = Modality(
                    latent=video_flat, timesteps=video_timesteps, positions=video_positions,
                    context=video_embeddings_pos, context_mask=None, enabled=True,
                    positional_embeddings=precomputed_video_rope,
                )
                audio_modality_pos = Modality(
                    latent=audio_flat, timesteps=audio_timesteps, positions=audio_positions,
                    context=audio_embeddings_pos, context_mask=None, enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                )
                video_vel_pos, audio_vel_pos = transformer(video=video_modality_pos, audio=audio_modality_pos)

                if use_cfg:
                    video_modality_neg = Modality(
                        latent=video_flat, timesteps=video_timesteps, positions=video_positions,
                        context=video_embeddings_neg, context_mask=None, enabled=True,
                        positional_embeddings=precomputed_video_rope,
                    )
                    audio_modality_neg = Modality(
                        latent=audio_flat, timesteps=audio_timesteps, positions=audio_positions,
                        context=audio_embeddings_neg, context_mask=None, enabled=True,
                        positional_embeddings=precomputed_audio_rope,
                    )
                    video_vel_neg, audio_vel_neg = transformer(video=video_modality_neg, audio=audio_modality_neg)
                    video_velocity_flat = video_vel_pos + (cfg_scale - 1.0) * (video_vel_pos - video_vel_neg)
                    audio_velocity_flat = audio_vel_pos + (cfg_scale - 1.0) * (audio_vel_pos - audio_vel_neg)
                else:
                    video_velocity_flat = video_vel_pos
                    audio_velocity_flat = audio_vel_pos

            video_velocity = mx.reshape(mx.transpose(video_velocity_flat, (0, 2, 1)), (b, c, f, h, w))
            audio_velocity = mx.reshape(audio_velocity_flat, (ab, at, ac, af))
            audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))

            video_denoised = to_denoised(video_latents_in, video_velocity, sigma_in)
            audio_denoised = to_denoised(audio_latents_in, audio_velocity, sigma_in)

            if video_state is not None:
                video_denoised = apply_denoise_mask(
                    video_denoised, video_state.clean_latent, video_state.denoise_mask
                )

            sigma_next_f32 = sigma_next_in.astype(mx.float32)
            sigma_f32 = sigma_in.astype(mx.float32)

            video_latents_f32 = video_latents_in.astype(mx.float32)
            video_denoised_f32 = video_denoised.astype(mx.float32)
            video_latents_out = (
                video_denoised_f32 + sigma_next_f32 * (video_latents_f32 - video_denoised_f32) / sigma_f32
            ).astype(dtype)

            audio_latents_f32 = audio_latents_in.astype(mx.float32)
            audio_denoised_f32 = audio_denoised.astype(mx.float32)
            audio_latents_out = (
                audio_denoised_f32 + sigma_next_f32 * (audio_latents_f32 - audio_denoised_f32) / sigma_f32
            ).astype(dtype)

            return video_latents_out, audio_latents_out

        step_fn = mx.compile(step_fn, shapeless=compile_shapeless)
    else:
        step_fn = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("[cyan]Denoising A/V (CFG)[/]", total=num_steps)
        progress_echo = os.environ.get("MLX_VIDEO_PROGRESS_ECHO") == "1"
        try:
            progress_echo_every = int(
                os.environ.get(
                    "MLX_VIDEO_DENOISE_ECHO_EVERY",
                    os.environ.get("MLX_VIDEO_PROGRESS_ECHO_EVERY", "12"),
                )
            )
        except Exception:
            progress_echo_every = 12
        last_echo_i = -1
        t0 = time.perf_counter()
        desc_plain = "Denoising"

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]
            sigma_mx = sigmas_mx[i]
            sigma_next_mx = sigmas_mx[i + 1]

            if step_fn is not None:
                video_latents, audio_latents = step_fn(video_latents, audio_latents, sigma_mx, sigma_next_mx)
            else:
                video_flat = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))

                # Flatten audio latents
                audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
                audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

                # Compute timesteps
                video_timesteps = sigma_mx * video_timesteps_mask
                audio_timesteps = sigma_mx * audio_timesteps_mask

                if cfg_batch:
                    video_latents_cat = mx.broadcast_to(mx.expand_dims(video_flat, axis=0), (2,) + video_flat.shape)
                    video_latents_cat = mx.reshape(video_latents_cat, (2 * b,) + video_flat.shape[1:])
                    audio_latents_cat = mx.broadcast_to(mx.expand_dims(audio_flat, axis=0), (2,) + audio_flat.shape)
                    audio_latents_cat = mx.reshape(audio_latents_cat, (2 * ab,) + audio_flat.shape[1:])
                    video_timesteps_cat = mx.broadcast_to(mx.expand_dims(video_timesteps, axis=0), (2,) + video_timesteps.shape)
                    video_timesteps_cat = mx.reshape(video_timesteps_cat, (2 * b,) + video_timesteps.shape[1:])
                    audio_timesteps_cat = mx.broadcast_to(mx.expand_dims(audio_timesteps, axis=0), (2,) + audio_timesteps.shape)
                    audio_timesteps_cat = mx.reshape(audio_timesteps_cat, (2 * ab,) + audio_timesteps.shape[1:])

                    video_modality = Modality(
                        latent=video_latents_cat, timesteps=video_timesteps_cat, positions=video_positions_cfg,
                        context=video_context_cat, context_mask=None, enabled=True,
                        positional_embeddings=precomputed_video_rope_cfg,
                    )
                    audio_modality = Modality(
                        latent=audio_latents_cat, timesteps=audio_timesteps_cat, positions=audio_positions_cfg,
                        context=audio_context_cat, context_mask=None, enabled=True,
                        positional_embeddings=precomputed_audio_rope_cfg,
                    )
                    video_vel_cat, audio_vel_cat = transformer(video=video_modality, audio=audio_modality)
                    video_vel_pos, video_vel_neg = mx.split(video_vel_cat, 2, axis=0)
                    audio_vel_pos, audio_vel_neg = mx.split(audio_vel_cat, 2, axis=0)
                    video_velocity_flat = video_vel_pos + (cfg_scale - 1.0) * (video_vel_pos - video_vel_neg)
                    audio_velocity_flat = audio_vel_pos + (cfg_scale - 1.0) * (audio_vel_pos - audio_vel_neg)
                else:
                    # Positive conditioning pass
                    video_modality_pos = Modality(
                        latent=video_flat, timesteps=video_timesteps, positions=video_positions,
                        context=video_embeddings_pos, context_mask=None, enabled=True,
                        positional_embeddings=precomputed_video_rope,
                    )
                    audio_modality_pos = Modality(
                        latent=audio_flat, timesteps=audio_timesteps, positions=audio_positions,
                        context=audio_embeddings_pos, context_mask=None, enabled=True,
                        positional_embeddings=precomputed_audio_rope,
                    )
                    video_vel_pos, audio_vel_pos = transformer(video=video_modality_pos, audio=audio_modality_pos)

                    if use_cfg:
                        # Negative conditioning pass
                        video_modality_neg = Modality(
                            latent=video_flat, timesteps=video_timesteps, positions=video_positions,
                            context=video_embeddings_neg, context_mask=None, enabled=True,
                            positional_embeddings=precomputed_video_rope,
                        )
                        audio_modality_neg = Modality(
                            latent=audio_flat, timesteps=audio_timesteps, positions=audio_positions,
                            context=audio_embeddings_neg, context_mask=None, enabled=True,
                            positional_embeddings=precomputed_audio_rope,
                        )
                        video_vel_neg, audio_vel_neg = transformer(video=video_modality_neg, audio=audio_modality_neg)

                        # Apply CFG
                        video_velocity_flat = video_vel_pos + (cfg_scale - 1.0) * (video_vel_pos - video_vel_neg)
                        audio_velocity_flat = audio_vel_pos + (cfg_scale - 1.0) * (audio_vel_pos - audio_vel_neg)
                    else:
                        video_velocity_flat = video_vel_pos
                        audio_velocity_flat = audio_vel_pos

                # Reshape velocities
                video_velocity = mx.reshape(mx.transpose(video_velocity_flat, (0, 2, 1)), (b, c, f, h, w))
                audio_velocity = mx.reshape(audio_velocity_flat, (ab, at, ac, af))
                audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))

                # Compute denoised
                video_denoised = to_denoised(video_latents, video_velocity, sigma_mx)
                audio_denoised = to_denoised(audio_latents, audio_velocity, sigma_mx)

                if video_state is not None:
                    video_denoised = apply_denoise_mask(
                        video_denoised, video_state.clean_latent, video_state.denoise_mask
                    )

                # Euler step
                if sigma_next > 0:
                    # Compute Euler step in float32 for precision (matching PyTorch behavior)
                    sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                    sigma_f32 = mx.array(sigma, dtype=mx.float32)

                    video_latents_f32 = video_latents.astype(mx.float32)
                    video_denoised_f32 = video_denoised.astype(mx.float32)
                    video_latents = (
                        video_denoised_f32 + sigma_next_f32 * (video_latents_f32 - video_denoised_f32) / sigma_f32
                    ).astype(dtype)

                    audio_latents_f32 = audio_latents.astype(mx.float32)
                    audio_denoised_f32 = audio_denoised.astype(mx.float32)
                    audio_latents = (
                        audio_denoised_f32 + sigma_next_f32 * (audio_latents_f32 - audio_denoised_f32) / sigma_f32
                    ).astype(dtype)
                else:
                    video_latents = video_denoised
                    audio_latents = audio_denoised

            if (i + 1) % eval_interval == 0 or i == num_steps - 1:
                mx.eval(video_latents, audio_latents)
            progress.advance(task)
            done = i + 1
            elapsed = time.perf_counter() - t0
            eta_s = (elapsed / max(1, done)) * max(0, num_steps - done)

            if progress_echo and progress_echo_every > 0:
                if i == 0 or i == num_steps - 1 or (i - last_echo_i) >= progress_echo_every:
                    print(f"{desc_plain} {done}/{num_steps} ETA {_format_eta(eta_s)}", flush=True)
                    last_echo_i = i

            _ui_event(
                {
                    "kind": "progress",
                    "phase": ui_phase,
                    "current": done,
                    "total": num_steps,
                    "percent": 100.0 * done / max(1, num_steps),
                    "eta_seconds": float(eta_s),
                }
            )

    _debug_stats("video_latents_after_denoise_dev_av", video_latents)
    _debug_stats("audio_latents_after_denoise_dev_av", audio_latents)
    return video_latents, audio_latents


# =============================================================================
# Audio Loading and Processing
# =============================================================================

def load_audio_decoder(model_path: Path, pipeline: PipelineType, unified_weights: Optional[dict] = None):
    """Load audio VAE decoder."""
    from mlx_video.models.ltx.audio_vae import AudioDecoder, CausalityAxis, NormType
    from mlx_video.convert import sanitize_audio_vae_weights

    decoder = AudioDecoder(
        ch=128,
        out_ch=2,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=set(),
        resolution=256,
        z_channels=AUDIO_LATENT_CHANNELS,
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=64,
        mid_block_add_attention=False,  # Config says no attention in mid block
    )

    raw_weights = None
    from_unified = False
    if unified_weights is not None:
        unified_audio = {
            k[len("audio_vae."):]: v
            for k, v in unified_weights.items()
            if k.startswith("audio_vae.")
        }
        if unified_audio:
            raw_weights = unified_audio
            from_unified = True

    if raw_weights is None:
        weight_file = model_path / (
            "ltx-2-19b-dev.safetensors" if pipeline == PipelineType.DEV else "ltx-2-19b-distilled.safetensors"
        )
        audio_vae_file = model_path / "audio_vae" / "diffusion_pytorch_model.safetensors"
        if audio_vae_file.exists():
            raw_weights = mx.load(str(audio_vae_file))
        elif weight_file.exists():
            raw_weights = mx.load(str(weight_file))
        else:
            raise FileNotFoundError(
                f"Audio VAE weights not found in {model_path}. "
                "Include audio_vae/diffusion_pytorch_model.safetensors or full base weights in the same repo."
            )

    sanitized = raw_weights if from_unified else sanitize_audio_vae_weights(raw_weights)
    if sanitized:
        # strip encoder prefix for decoder
        dec_weights = {k.replace("decoder.", ""): v for k, v in sanitized.items() if k.startswith("decoder.")}
        decoder.load_weights(list(dec_weights.items()), strict=False)
        if "per_channel_statistics._mean_of_means" in sanitized:
            decoder.per_channel_statistics._mean_of_means = sanitized["per_channel_statistics._mean_of_means"]
        if "per_channel_statistics._std_of_means" in sanitized:
            decoder.per_channel_statistics._std_of_means = sanitized["per_channel_statistics._std_of_means"]

    return decoder


def load_vocoder(model_path: Path, pipeline: PipelineType, unified_weights: Optional[dict] = None):
    """Load vocoder for mel to waveform conversion."""
    from mlx_video.models.ltx.audio_vae import Vocoder
    from mlx_video.convert import sanitize_vocoder_weights

    vocoder = Vocoder(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[6, 5, 2, 2, 2],
        upsample_kernel_sizes=[16, 15, 8, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1024,
        stereo=True,
        output_sample_rate=AUDIO_SAMPLE_RATE,
    )

    raw_weights = None
    from_unified = False
    if unified_weights is not None:
        unified_vocoder = {
            k[len("vocoder."):]: v
            for k, v in unified_weights.items()
            if k.startswith("vocoder.")
        }
        if unified_vocoder:
            raw_weights = unified_vocoder
            from_unified = True

    if raw_weights is None:
        weight_file = model_path / (
            "ltx-2-19b-dev.safetensors" if pipeline == PipelineType.DEV else "ltx-2-19b-distilled.safetensors"
        )
        vocoder_file = model_path / "vocoder" / "diffusion_pytorch_model.safetensors"
        if vocoder_file.exists():
            raw_weights = mx.load(str(vocoder_file))
        elif weight_file.exists():
            raw_weights = mx.load(str(weight_file))
        else:
            raise FileNotFoundError(
                f"Vocoder weights not found in {model_path}. "
                "Include vocoder/diffusion_pytorch_model.safetensors or full base weights in the same repo."
            )

    sanitized = raw_weights if from_unified else sanitize_vocoder_weights(raw_weights)
    if sanitized:
        vocoder.load_weights(list(sanitized.items()), strict=False)

    return vocoder


def _write_video_cv2(video_np: np.ndarray, path: Path, fps: float, console: Optional[Console] = None) -> None:
    """Write a uint8 RGB video to disk via OpenCV with codec fallback."""
    import cv2

    h, w = video_np.shape[1], video_np.shape[2]
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if out.isOpened():
            for frame in video_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            return
        out.release()
    if console:
        console.print(f"[red]❌ Could not open video writer for {path}[/]")
    raise RuntimeError(f"Could not open video writer for {path}")


def _write_video_ffmpeg(
    video_np: np.ndarray,
    path: Path,
    fps: float,
    console: Optional[Console] = None,
    codec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 18,
) -> None:
    """Write a uint8 RGB video to disk via ffmpeg (raw RGB24 pipe).

    This can be significantly faster than OpenCV encoding and often yields better
    quality/bitrate control. Falls back to callers when ffmpeg isn't available.
    """
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found")

    h, w = video_np.shape[1], video_np.shape[2]
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]

    try:
        # Use DEVNULL for stdout to avoid deadlocks on long runs; stderr is enough for diagnostics.
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        # Avoid materializing a giant bytes object; stream frame bytes.
        for frame in video_np:
            proc.stdin.write(frame.tobytes(order="C"))
        proc.stdin.close()
        # Do not call communicate() after closing stdin; it may attempt to flush a closed file.
        proc.wait()
        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="ignore"))
    except Exception as exc:
        if console:
            console.print(f"[yellow]⚠️  ffmpeg encode failed; falling back. ({exc})[/]")
        raise


def _write_video(
    video_np: np.ndarray,
    path: Path,
    fps: float,
    console: Optional[Console] = None,
    encoder: str = "cv2",
) -> None:
    """Write video using the selected encoder ("cv2" or "ffmpeg")."""
    if encoder == "ffmpeg":
        try:
            _write_video_ffmpeg(video_np, path, fps, console=console)
            return
        except Exception:
            # If ffmpeg is unavailable or fails, fall back to cv2 so generation can
            # still complete (even if quality controls are reduced).
            if console:
                console.print("[yellow]⚠️  Falling back to OpenCV video encoding.[/]")
    _write_video_cv2(video_np, path, fps, console=console)


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio to WAV file."""
    import wave

    if audio.ndim == 2:
        audio = audio.T

    # Clamp and sanitize to avoid NaN/Inf propagating into the WAV
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Combine video and audio into final output using ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        # Explicit mapping avoids ffmpeg picking the wrong streams when inputs
        # contain unexpected metadata/streams.
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        # Do NOT use -shortest here. Audio generation may be slightly shorter
        # than the video duration due to hop-size rounding, and -shortest will
        # truncate the video (dropping last frames).
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg error: {e.stderr.decode()}[/]")
        return False
    except FileNotFoundError:
        console.print("[red]FFmpeg not found. Please install ffmpeg.[/]")
        return False


# =============================================================================
# Unified Generate Function
# =============================================================================

def generate_video(
    model_repo: str,
    text_encoder_repo: str,
    prompt: str,
    pipeline: PipelineType = PipelineType.DISTILLED,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    num_inference_steps: int = 40,
    cfg_scale: float = 4.0,
    seed: int = 42,
    fps: float = 24.0,
    output_path: str = "output.mp4",
    save_frames: bool = False,
    video_encoder: str = "cv2",
    verbose: bool = True,
    profile: bool = False,
    profile_json_path: Optional[str] = None,
    enhance_prompt: bool = False,
    enhance_prompt_model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    images: Optional[list[tuple[str, int, float]]] = None,
    video_conditionings: Optional[list[tuple[str, int, float]]] = None,
    distilled_loras: Optional[list[tuple[str, float]]] = None,
    conditioning_mode: str = "replace",
    tiling: str = "auto",
    stream: bool = False,
    audio: bool = False,
    audio_mode: str = "auto",
    audio_model_repo: Optional[str] = None,
    audio_steps: int = 8,
    output_audio_path: Optional[str] = None,
    mem_log: bool = False,
    clear_cache: bool = False,
    cache_limit_gb: Optional[float] = None,
    memory_limit_gb: Optional[float] = None,
    eval_interval: int = 1,
    compile_step: bool = False,
    compile_shapeless: bool = False,
    cfg_batch: bool = False,
    fp32_euler: bool = True,
    metal_capture: bool = False,
    metal_capture_path: Optional[str] = None,
    metal_capture_phase: str = "denoise",
    loras: Optional[list[tuple[str, float]]] = None,
    checkpoint_path: Optional[str] = None,
    stage2_model_repo: Optional[str] = None,
    stage2_dev: bool = False,
    stage1_steps: int = 8,
    stage2_steps: int = 3,
    sigma_subsample: str = "farthest",
    auto_output_name: bool = False,
    output_name_model: Optional[str] = None,
):
    """Generate video using LTX-2 models.

    Supports two pipelines:
    - DISTILLED: Two-stage generation with upsampling, fixed sigma schedules, no CFG
    - DEV: Single-stage generation with dynamic sigmas and CFG

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        pipeline: Pipeline type (DISTILLED or DEV)
        negative_prompt: Negative prompt for CFG (dev pipeline only)
        height: Output video height (must be divisible by 32/64)
        width: Output video width (must be divisible by 32/64)
        num_frames: Number of frames (must be 1 + 8*k)
        num_inference_steps: Number of denoising steps (dev pipeline only)
        cfg_scale: Guidance scale for CFG (dev pipeline only)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        save_frames: Whether to save individual frames as images
        video_encoder: Video encoder to use for MP4 writing ("cv2" or "ffmpeg")
        verbose: Whether to print progress
        profile: Print a phase timing breakdown at the end (for benchmarking)
        profile_json_path: Optional path to write phase timings as JSON ("auto" writes next to output)
        enhance_prompt: Whether to enhance prompt using Gemma
        enhance_prompt_model: Optional model repo for prompt enhancement
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V (single)
        image_strength: Conditioning strength for I2V (single)
        image_frame_idx: Frame index to condition for I2V (single)
        images: List of conditioning images (path, frame_idx, strength)
        video_conditionings: List of video conditionings (path, frame_idx, strength)
        distilled_loras: LoRAs to apply for stage-2 refinement (distilled pipeline)
        conditioning_mode: "replace" (default) or "guide" (keyframe-style)
        tiling: Tiling mode for VAE decoding
        stream: Stream frames to output as they're decoded
        audio: Enable synchronized audio generation
        audio_mode: Audio generation strategy when --audio is enabled:
            - auto: use "separate" for distilled pipelines (better audio), "joint" for dev
            - joint: generate audio latents together with video (AudioVideo model)
            - separate: generate video first (VideoOnly), then audio with an AudioOnly model
        audio_model_repo: Optional model repo/directory for separate audio transformer weights.
        audio_steps: Number of denoising steps for separate audio generation (default 8).
        output_audio_path: Path to save audio file
        mem_log: Log active/cache/peak memory at key stages
        clear_cache: Clear MLX cache after generation
        cache_limit_gb: Set MLX cache limit in GB
        memory_limit_gb: Set MLX memory limit in GB
        eval_interval: Evaluate latents every N steps (reduces sync overhead)
        compile_step: Compile denoise step for repeated execution
        compile_shapeless: Allow recompilation for varying shapes (slower, more flexible)
        cfg_batch: If True, batch CFG pos/neg into one forward (more memory)
        fp32_euler: If True, compute Euler updates in float32 for numerical stability (slower)
        metal_capture: Enable Metal GPU capture (requires MTL_CAPTURE_ENABLED=1)
        metal_capture_path: Path for the .gputrace output (must not exist)
        metal_capture_phase: "denoise", "decode", or "all"
        loras: Optional list of (path, strength) LoRA weights to merge
        checkpoint_path: Optional explicit checkpoint .safetensors file to load
        stage2_model_repo: Optional model repo/directory for stage-2 refinement (distilled pipelines only)
        stage2_dev: If True, use the dev CFG denoiser for stage-2 refinement (distilled pipelines only)
        stage1_steps: Number of denoising steps in stage 1 (distilled pipelines only; default 8)
        stage2_steps: Number of refinement steps in stage 2 (distilled pipelines only; default 3)
        sigma_subsample: Subsampling method for fixed sigma schedules when reducing steps ("uniform" or "farthest")
        auto_output_name: If True, auto-generate output filename from prompt
        output_name_model: Optional model repo for filename generation
    """
    # Track stage-specific compile behavior for profiling/debugging.
    stage1_compile_effective: Optional[bool] = None
    stage2_compile_effective: Optional[bool] = None

    start_time = time.time()
    phase_timer = _PhaseTimer(profile)
    if video_encoder not in ("cv2", "ffmpeg"):
        console.print(f"[yellow]⚠️  Unknown video encoder '{video_encoder}', using cv2[/]")
        video_encoder = "cv2"
    capture_path = None
    capture_phase = metal_capture_phase
    capture_started = False
    if metal_capture:
        if metal_capture_path:
            capture_path = Path(metal_capture_path)
        else:
            capture_path = Path(output_path).with_suffix(".gputrace")
        if capture_phase not in ("denoise", "decode", "all"):
            console.print("[yellow]⚠️  Invalid metal capture phase; using 'denoise'.[/]")
            capture_phase = "denoise"
    if cache_limit_gb is not None:
        limit_bytes = int(cache_limit_gb * (1024 ** 3))
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(limit_bytes)
        elif hasattr(mx, "metal"):
            mx.metal.set_cache_limit(limit_bytes)
    if memory_limit_gb is not None:
        limit_bytes = int(memory_limit_gb * (1024 ** 3))
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(limit_bytes)
        elif hasattr(mx, "metal"):
            mx.metal.set_memory_limit(limit_bytes)
    _log_memory("start", mem_log)

    # Normalize conditioning inputs
    images_list = list(images or [])
    if image is not None:
        images_list.append((image, image_frame_idx, image_strength))
    normalized_images = []
    for img_path, frame_idx, strength in images_list:
        if frame_idx is None:
            frame_idx = 0
        if strength is None:
            strength = 1.0
        normalized_images.append((img_path, int(frame_idx), float(strength)))
    images_list = normalized_images

    raw_video_conditionings = list(video_conditionings or [])
    video_conditionings = []
    for item in raw_video_conditionings:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            path, strength = item
            frame_idx = 0
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            path, frame_idx, strength = item
        else:
            raise ValueError(f"Invalid video conditioning entry: {item}")
        video_conditionings.append((path, int(frame_idx), float(strength)))

    # Handle extended pipeline types
    if pipeline == PipelineType.KEYFRAME:
        conditioning_mode = "guide"
    if pipeline == PipelineType.IC_LORA:
        if not video_conditionings:
            raise ValueError("IC-LoRA pipeline requires --video-conditioning PATH [FRAME_IDX] STRENGTH")
        conditioning_mode = "replace"

    is_distilled_pipeline = pipeline in (PipelineType.DISTILLED, PipelineType.KEYFRAME, PipelineType.IC_LORA)
    is_dev_pipeline = pipeline == PipelineType.DEV

    if is_dev_pipeline and video_conditionings:
        raise ValueError("Video conditioning is only supported in ic_lora/distilled pipelines.")

    # Validate dimensions
    divisor = 64 if is_distilled_pipeline else 32
    # Distilled-style pipelines require height/width divisible by 64 (dev requires 32).
    # Instead of failing hard, pad and crop back so the UI can request e.g. 832x480.
    output_height, output_width = height, width
    crop_params: tuple[int, int, int, int] | None = None  # (top, left, out_h, out_w)
    if height % divisor != 0 or width % divisor != 0:
        pad_h = (divisor - (height % divisor)) % divisor
        pad_w = (divisor - (width % divisor)) % divisor
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        crop_params = (pad_top, pad_left, output_height, output_width)
        height = height + pad_top + pad_bottom
        width = width + pad_left + pad_right

        if verbose:
            console.print(
                f"[yellow]⚠️  {pipeline.value} requires dims divisible by {divisor}; "
                f"padding to {width}x{height} and cropping output back to {output_width}x{output_height}.[/]"
            )

    if num_frames % 8 != 1:
        # Always round up to avoid shortening the requested duration
        adjusted_num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1
        if verbose:
            console.print(f"[dim]Adjusted num_frames to {adjusted_num_frames} (1 + 8*k requirement).[/]")
        num_frames = adjusted_num_frames

    is_i2v = len(images_list) > 0
    mode_str = "I2V" if is_i2v else "T2V"
    if sigma_subsample not in ("uniform", "farthest"):
        raise ValueError("--sigma-subsample must be 'uniform' or 'farthest'.")

    audio_mode = (audio_mode or "auto").lower()
    if audio_mode not in ("auto", "joint", "separate"):
        raise ValueError("--audio-mode must be one of: auto, joint, separate")

    audio_mode_effective = "off"
    joint_audio = False
    separate_audio = False
    if audio:
        if audio_mode == "auto":
            # Distilled audio latents are often tonal; separate audio uses a dev
            # AudioOnly transformer and tends to be less "zoom"-like.
            audio_mode_effective = "separate" if pipeline != PipelineType.DEV else "joint"
        else:
            audio_mode_effective = audio_mode
        joint_audio = audio_mode_effective == "joint"
        separate_audio = audio_mode_effective == "separate"
        mode_str += "+Audio"

    if pipeline == PipelineType.DEV:
        pipeline_name = "DEV"
    elif pipeline == PipelineType.KEYFRAME:
        pipeline_name = "KEYFRAME"
    elif pipeline == PipelineType.IC_LORA:
        pipeline_name = "IC_LORA"
    else:
        pipeline_name = "DISTILLED"
    header = f"[bold cyan]🎬 [{pipeline_name}] [{mode_str}] {output_width}x{output_height} • {num_frames} frames[/]"
    console.print(Panel(header, expand=False))
    console.print(f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/]")

    if is_dev_pipeline:
        console.print(f"[dim]Steps: {num_inference_steps}, CFG: {cfg_scale}[/]")
    if audio:
        console.print(f"[dim]Audio mode: {audio_mode_effective}[/]")

    if is_i2v:
        for img_path, frame_idx, strength in images_list:
            console.print(f"[dim]Image: {img_path} (strength={strength}, frame={frame_idx})[/]")
        if conditioning_mode == "guide":
            console.print("[dim]Image conditioning mode: guide (keyframe)[/]")

    if video_conditionings:
        for vpath, vframe_idx, vstrength in video_conditionings:
            console.print(f"[dim]Video conditioning: {vpath} (frame={vframe_idx}, strength={vstrength})[/]")

    if loras:
        console.print(f"[dim]LoRAs: {len(loras)}[/]")
        for lora_path, strength in loras:
            console.print(f"[dim]  - {lora_path} (strength={strength})[/]")

    audio_frames = None
    if audio:
        audio_frames = compute_audio_frames(num_frames, fps)
        console.print(f"[dim]Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz[/]")

    output_path = Path(output_path)

    # Runtime quantization: used to avoid downloading huge/broken "pre-quantized" snapshots
    # and to reduce memory/speed costs. Defaults match the common AITRADER 4/8-bit conversions.
    runtime_quantize = False
    runtime_quantize_bits = int(os.environ.get("LTX_RUNTIME_QUANT_BITS", "8"))
    runtime_quantize_group_size = int(os.environ.get("LTX_RUNTIME_QUANT_GROUP_SIZE", "64"))
    runtime_quantize_mode = str(os.environ.get("LTX_RUNTIME_QUANT_MODE", "affine"))
    runtime_quantize_scope = str(os.environ.get("LTX_RUNTIME_QUANT_SCOPE", "video_core"))
    prefetched_quant_meta: dict | None = None

    # Get model path (HF repo or local directory)
    explicit_weight_path = None
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        if candidate.is_dir():
            model_path = candidate
        else:
            model_path = candidate.parent
            explicit_weight_path = candidate
    else:
        # Avoid downloading huge "pre-quantized" snapshots up-front. For known
        # 4/8-bit preset repos we only need `quantization.json` to decide whether
        # to use the snapshot directly or fall back to runtime quantization.
        #
        # This keeps first-run installs responsive and prevents wasting bandwidth
        # on broken BF16-quantized community repos.
        import re

        repo_str = str(model_repo)
        preset_match = re.search(r"ltx2-(?:dev|distilled)-(4|8)bit-mlx$", repo_str)
        if preset_match and not Path(repo_str).expanduser().exists():
            prefetched_qmeta = None
            try:
                import json
                from huggingface_hub import hf_hub_download

                qmeta_file = hf_hub_download(repo_id=repo_str, filename="quantization.json")
                with open(qmeta_file, "r") as f:
                    prefetched_qmeta = json.load(f)
                _debug_log(f"prefetched quantization.json for {repo_str}: {prefetched_qmeta}")
            except Exception as exc:
                _debug_log(f"prefetch quantization.json failed for {repo_str}: {exc}")

            # Default to runtime quantization for known 4/8-bit preset repos.
            #
            # Rationale:
            # - Some community "pre-quantized" snapshots are enormous and/or yield severe
            #   artifacts ("snow"/static). Runtime quantization uses a known-good BF16 base
            #   model and quantizes deterministically on the user machine.
            #
            # Users can opt back into the snapshot by setting `LTX_USE_PREQUANT=1`.
            use_prequant = os.environ.get("LTX_USE_PREQUANT", "").lower() in ("1", "true", "yes")
            force_runtime = os.environ.get("LTX_FORCE_RUNTIME_QUANT", "").lower() in ("1", "true", "yes")
            use_runtime = force_runtime or (not use_prequant)

            if use_runtime:
                runtime_quantize = True
                runtime_quantize_bits = int(preset_match.group(1))
                if prefetched_qmeta:
                    prefetched_quant_meta = prefetched_qmeta
                    runtime_quantize_group_size = int(prefetched_qmeta.get("group_size", runtime_quantize_group_size))
                    runtime_quantize_mode = str(prefetched_qmeta.get("mode", runtime_quantize_mode))
                    runtime_quantize_scope = str(prefetched_qmeta.get("quantize_scope", runtime_quantize_scope))
                    # Some conversions used "core" to mean "video core" (not including audio blocks).
                    if runtime_quantize_scope == "core":
                        runtime_quantize_scope = "video_core"

                base_repo = os.environ.get(
                    "LTX_RUNTIME_QUANT_BASE_REPO",
                    "mlx-community/LTX-2-distilled-bf16" if is_distilled_pipeline else "mlx-community/LTX-2-dev-bf16",
                )
                _debug_log(
                    f"runtime_quantize(preset): repo={repo_str} "
                    f"bits={runtime_quantize_bits} group_size={runtime_quantize_group_size} "
                    f"mode={runtime_quantize_mode} scope={runtime_quantize_scope} base_repo={base_repo}"
                )
                model_path = get_model_path(base_repo)
            else:
                _debug_log(f"using pre-quantized snapshot as-is: repo={repo_str}")
                model_path = get_model_path(model_repo)
        else:
            model_path = get_model_path(model_repo)

    _debug_log(
        f"pipeline={pipeline.value} model_repo={model_repo} model_path={model_path} "
        f"checkpoint_path={checkpoint_path}"
    )

    # Default to the official LTX-2 repo for the text encoder weights. Some community
    # conversions (e.g. quantized snapshots) omit `text_encoder/` entirely and rely on
    # this fallback. Using an unrelated LLM repo here can cause huge downloads and
    # unstable "snow"/garbled generations.
    default_text_encoder = os.environ.get("LTX_TEXT_ENCODER_REPO", "Lightricks/LTX-2")
    if text_encoder_repo is None:
        # Prefer bundled text_encoder folder if present; otherwise fall back to default repo.
        if (model_path / "text_encoder").is_dir():
            text_encoder_path = model_path
        else:
            text_encoder_repo = default_text_encoder
            text_encoder_path = get_model_path(text_encoder_repo, require_files=False)
    else:
        text_encoder_path = get_model_path(text_encoder_repo, require_files=False)

    _debug_log(f"text_encoder_repo={text_encoder_repo} text_encoder_path={text_encoder_path}")
    quant_meta = None
    preferred_dtype: mx.Dtype | None = None

    quant_meta_path = model_path / "quantization.json"
    if quant_meta_path.exists():
        try:
            import json
            with quant_meta_path.open("r") as f:
                quant_meta = json.load(f)
            _debug_log(f"quantization.json={quant_meta}")
        except Exception as exc:
            _debug_log(f"quantization.json read error: {exc}")
    if quant_meta is None and runtime_quantize and prefetched_quant_meta is not None:
        quant_meta = prefetched_quant_meta
        _debug_log(f"quantization.json (prefetched)={quant_meta}")
    if quant_meta is None and runtime_quantize:
        # Base BF16 repos usually do not ship quantization metadata. Ensure we still
        # pin compute dtype for stable runtime quantization.
        quant_meta = {"dtype": "bfloat16"}
    if quant_meta:
        q_dtype = str(quant_meta.get("dtype", "")).lower().strip()
        if q_dtype in ("bf16", "bfloat16"):
            preferred_dtype = mx.bfloat16
        elif q_dtype in ("f16", "float16", "fp16"):
            preferred_dtype = mx.float16
        elif q_dtype in ("f32", "float32", "fp32"):
            preferred_dtype = mx.float32

    # Auto-generate a descriptive output name from the prompt (optional)
    if auto_output_name:
        system_prompt = (
            "Return a short, filesystem-safe filename (3-8 words) describing the scene. "
            "Use only lowercase letters and spaces; no punctuation, no quotes. "
            "Return only the filename text, nothing else."
        )
        try:
            from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
            name_encoder = LTX2TextEncoder()
            name_model = output_name_model or text_encoder_repo
            name_encoder.load(model_path=model_path, text_encoder_path=name_model)
            mx.eval(name_encoder.parameters())
            raw_name = name_encoder.enhance_t2v(
                prompt,
                max_tokens=32,
                system_prompt=system_prompt,
                seed=seed,
                verbose=verbose,
                temperature=0.2,
            )
            del name_encoder
            mx.clear_cache()
        except Exception:
            raw_name = prompt

        safe_name = _slugify_filename(raw_name)
        out_dir = output_path if output_path.suffix == "" else output_path.parent
        suffix = output_path.suffix if output_path.suffix else ".mp4"
        output_path = out_dir / f"{safe_name}{suffix}"

    # Model weight file (base PyTorch weights) and optional MLX-converted transformer.
    #
    # Some community repos (including the AITRADER 4/8-bit snapshots) may ship multiple
    # variants (fp16 + quant) under different filenames. Prefer the most specific match
    # to avoid accidentally loading the wrong checkpoint (which can manifest as "snow"
    # artifacts when LoRAs trigger re-quantization).
    weight_file = "ltx-2-19b-dev.safetensors" if is_dev_pipeline else "ltx-2-19b-distilled.safetensors"
    weight_file_path = explicit_weight_path or (model_path / weight_file)
    model_kind = "dev" if is_dev_pipeline else "distilled"
    repo_hint = f"{model_repo}".lower()
    bits_hint: str | None = None
    if any(x in repo_hint for x in ("8bit", "q8", "int8")):
        bits_hint = "8bit"
    elif any(x in repo_hint for x in ("4bit", "q4", "int4")):
        bits_hint = "4bit"

    mlx_weight_candidates: list[Path] = []
    if bits_hint:
        mlx_weight_candidates.append(model_path / f"ltx-2-19b-{model_kind}-{bits_hint}-mlx.safetensors")
    mlx_weight_candidates.append(model_path / f"ltx-2-19b-{model_kind}-mlx.safetensors")
    mlx_weight_file = next((p for p in mlx_weight_candidates if p.exists()), mlx_weight_candidates[-1])
    transformer_weight_path = explicit_weight_path or (
        mlx_weight_file if mlx_weight_file.exists() else model_path / weight_file
    )

    _debug_log(f"weight_file={weight_file} weight_file_path={weight_file_path}")
    _debug_log(f"mlx_weight_file={mlx_weight_file} transformer_weight_path={transformer_weight_path}")
    _debug_weights_summary("transformer_weights", transformer_weight_path)
    if weight_file_path != transformer_weight_path:
        _debug_weights_summary("base_weights", weight_file_path)

    unified_weights = None
    unified_path = model_path / "model.safetensors"
    if unified_path.exists() and os.environ.get("LTX_DISABLE_UNIFIED") != "1":
        try:
            unified_weights = mx.load(str(unified_path))
            if not any(k.startswith("transformer.") for k in unified_weights):
                unified_weights = None
        except Exception as exc:
            _debug_log(f"unified load error: {exc}")
            unified_weights = None

    def _unified_subset(prefix: str) -> Optional[dict]:
        if unified_weights is None:
            return None
        subset = {
            k[len(prefix):]: v
            for k, v in unified_weights.items()
            if k.startswith(prefix)
        }
        return subset or None

    # Calculate latent dimensions
    if is_distilled_pipeline:
        stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
        stage2_h, stage2_w = height // 32, width // 32
    else:
        latent_h, latent_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    def _resolve_frame_idx(frame_idx: int) -> int:
        """Map a video-frame index to a latent-frame index (safe for small clips)."""
        if frame_idx < latent_frames:
            return frame_idx
        if num_frames <= 1 or latent_frames <= 1:
            return 0
        scaled = int((frame_idx / (num_frames - 1) * (latent_frames - 1)) + 0.5)
        return int(max(0, min(latent_frames - 1, scaled)))

    mx.random.seed(seed)

    def _resolve_vae_source() -> Path:
        """Resolve a VAE source path for encoding/decoding.

        Prefer the dedicated VAE weights shipped with the selected model repo. Some
        conversions (notably quantized snapshots) previously triggered a fallback to
        the full base LTX-2 checkpoints, which dramatically increases download size
        and can result in incorrect decoding (e.g. "snow"/static) if the chosen
        fallback does not actually contain VAE weights.
        """
        # Debug/ops override: force VAE from a specific local path or HF repo.
        # This is useful to isolate "snow"/static issues to either transformer latents
        # or VAE decoding mismatches.
        force_vae_path = os.environ.get("LTX_FORCE_VAE_PATH")
        if force_vae_path:
            try:
                forced = Path(force_vae_path).expanduser()
                if forced.exists():
                    _debug_log(f"vae_source=force_path {forced}")
                    return forced
            except Exception as exc:
                _debug_log(f"vae_source force_path error: {exc}")

        force_vae_repo = os.environ.get("LTX_FORCE_VAE_REPO")
        if force_vae_repo:
            try:
                forced_root = Path(get_model_path(force_vae_repo, require_files=False))
                forced_repo_vae = forced_root / "vae" / "diffusion_pytorch_model.safetensors"
                if forced_repo_vae.exists() and forced_repo_vae.stat().st_size > 0:
                    _debug_log(f"vae_source=force_repo_vae repo={force_vae_repo} path={forced_root}")
                    return forced_root
                for wf in ("ltx-2-19b-dev.safetensors", "ltx-2-19b-distilled.safetensors"):
                    cand = forced_root / wf
                    if cand.exists() and cand.stat().st_size > 0:
                        _debug_log(f"vae_source=force_repo_weight repo={force_vae_repo} path={cand}")
                        return cand
            except Exception as exc:
                _debug_log(f"vae_source force_repo error: {exc}")

        if _unified_subset("vae_decoder.") is not None:
            _debug_log(f"vae_source=unified {unified_path}")
            return unified_path
        # Prefer dedicated VAE weights inside the selected model repo.
        repo_vae = model_path / "vae" / "diffusion_pytorch_model.safetensors"
        if repo_vae.exists():
            try:
                if repo_vae.stat().st_size > 8:
                    _debug_log(f"vae_source=repo_vae {model_path}")
                    return model_path
            except Exception:
                # If stat fails but the path exists, still attempt to load from it.
                _debug_log(f"vae_source=repo_vae (stat error) {model_path}")
                return model_path

            # Edge case: placeholder/empty VAE file. Allow an override (best-effort).
            override_repo = os.environ.get("LTX_VAE_REPO")
            if override_repo:
                try:
                    override_root = Path(get_model_path(override_repo, require_files=False))
                    override_vae = override_root / "vae" / "diffusion_pytorch_model.safetensors"
                    if override_vae.exists() and override_vae.stat().st_size > 8:
                        _debug_log(f"vae_source=override_repo {override_repo} {override_root}")
                        return override_root
                    override_weight = override_root / weight_file
                    if override_weight.exists() and override_weight.stat().st_size > 0:
                        _debug_log(f"vae_source=override_repo_weight {override_repo} {override_weight}")
                        return override_weight
                except Exception as exc:
                    _debug_log(f"vae_source override error: {exc}")

            # Optional: allow users to explicitly opt into using the base VAE from Lightricks.
            # This can improve decode quality for some third-party conversions, but it may
            # require downloading very large checkpoints. Keep it off by default.
            if os.environ.get("LTX_USE_BASE_VAE") == "1":
                try:
                    base_root = Path(get_model_path("Lightricks/LTX-2", require_files=False))
                    base_vae = base_root / "vae" / "diffusion_pytorch_model.safetensors"
                    if base_vae.exists() and base_vae.stat().st_size > 8:
                        _debug_log(f"vae_source=base_repo_vae {base_root}")
                        return base_root
                    base_weight_fallback = base_root / weight_file
                    if base_weight_fallback.exists() and base_weight_fallback.stat().st_size > 0:
                        _debug_log(f"vae_source=base_fallback {base_weight_fallback}")
                        return base_weight_fallback
                except Exception as exc:
                    _debug_log(f"vae_source base fallback error: {exc}")

            _debug_log(f"vae_source=repo_vae_empty {model_path}")
            return model_path

        # Prefer full (non-MLX) weights inside the same repo (contains VAE stats/decoder).
        # NOTE: Quantized MLX weight files (e.g. `*-mlx.safetensors`) often do NOT include
        # the VAE stats/decoder, so we only use this fallback when a dedicated `vae/` folder
        # is missing.
        base_weight = model_path / weight_file
        if base_weight.exists():
            _debug_log(f"vae_source=base_weight {base_weight}")
            return base_weight
        # If an explicit checkpoint is provided and isn't the quantized MLX file, use it.
        if explicit_weight_path and explicit_weight_path.exists():
            if not explicit_weight_path.name.endswith("-mlx.safetensors"):
                _debug_log(f"vae_source=explicit {explicit_weight_path}")
                return explicit_weight_path
        raise FileNotFoundError(
            "VAE weights not found in the selected model repo. "
            "Include either `vae/diffusion_pytorch_model.safetensors` "
            "or the full `ltx-2-19b-*.safetensors` in the same repo."
        )

    def _resolve_upsampler_weight(filename: str) -> Path:
        """Resolve distilled upsampler weights with a safe fallback.

        Some community snapshots contain zero-byte placeholders for these files.
        Prefer the current repo when it's a real file, otherwise fall back to the
        official LTX-2 repo (and other known-good snapshots) to avoid runtime read
        errors.
        """
        candidate = model_path / filename
        try:
            if candidate.exists() and candidate.stat().st_size > 8:
                _debug_log(f"upsampler_source=repo {candidate}")
                return candidate
        except Exception:
            if candidate.exists():
                _debug_log(f"upsampler_source=repo (stat error) {candidate}")
                return candidate

        for fallback_repo in (
            "Lightricks/LTX-2",
            "mlx-community/LTX-2-dev-bf16",
            "mlx-community/LTX-2-distilled-bf16",
        ):
            try:
                fb_root = Path(get_model_path(fallback_repo, require_files=False))
                fb = fb_root / filename
                if fb.exists() and fb.stat().st_size > 8:
                    _debug_log(f"upsampler_source=fallback repo={fallback_repo} path={fb}")
                    return fb
            except Exception:
                continue

        _debug_log(f"upsampler_source=missing {candidate}")
        return candidate

    enhanced_with_alt = False

    # Load text encoder for embeddings
    with phase_timer.phase("text_encoder_load"):
        with console.status("[blue]📝 Loading text encoder...[/]", spinner="dots"):
            from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
            text_encoder = LTX2TextEncoder()
            text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
            mx.eval(text_encoder.parameters())
        console.print("[green]✓[/] Text encoder loaded")

    # Optionally enhance the prompt (default path, if not already enhanced)
    if enhance_prompt and not enhanced_with_alt:
        with phase_timer.phase("prompt_enhance"):
            console.print("[bold magenta]✨ Enhancing prompt[/]")
            prompt = text_encoder.enhance_t2v(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, verbose=verbose)
            console.print(f"[dim]Enhanced prompt:[/]\n{prompt}")

    # Some distilled workflows (e.g. dev refinement in stage-2) need dev-style
    # pos/neg embeddings even when the stage-1 pipeline is distilled.
    use_stage2_dev = bool(stage2_dev and is_distilled_pipeline)
    need_dev_embeddings = is_dev_pipeline or use_stage2_dev

    # Encode prompts
    video_embeddings_pos = video_embeddings_neg = None
    audio_embeddings_pos = audio_embeddings_neg = None
    text_embeddings = None
    audio_embeddings = None

    with phase_timer.phase("prompt_encode"):
        if need_dev_embeddings:
            # Dev-style: positive + negative embeddings for CFG (used by dev pipeline,
            # and optionally by distilled stage-2 dev refinement).
            if audio:
                video_embeddings_pos, audio_embeddings_pos = text_encoder(prompt, return_audio_embeddings=True)
                video_embeddings_neg, audio_embeddings_neg = text_encoder(negative_prompt, return_audio_embeddings=True)
                if preferred_dtype is not None:
                    video_embeddings_pos = video_embeddings_pos.astype(preferred_dtype)
                    video_embeddings_neg = video_embeddings_neg.astype(preferred_dtype)
                    audio_embeddings_pos = audio_embeddings_pos.astype(preferred_dtype)
                    audio_embeddings_neg = audio_embeddings_neg.astype(preferred_dtype)
                    model_dtype = preferred_dtype
                else:
                    model_dtype = video_embeddings_pos.dtype

                mx.eval(video_embeddings_pos, video_embeddings_neg, audio_embeddings_pos, audio_embeddings_neg)
                _debug_stats("video_embeddings_pos", video_embeddings_pos)
                _debug_stats("video_embeddings_neg", video_embeddings_neg)
                _debug_stats("audio_embeddings_pos", audio_embeddings_pos)
                _debug_stats("audio_embeddings_neg", audio_embeddings_neg)
            else:
                video_embeddings_pos, _ = text_encoder(prompt, return_audio_embeddings=False)
                video_embeddings_neg, _ = text_encoder(negative_prompt, return_audio_embeddings=False)
                if preferred_dtype is not None:
                    video_embeddings_pos = video_embeddings_pos.astype(preferred_dtype)
                    video_embeddings_neg = video_embeddings_neg.astype(preferred_dtype)
                    model_dtype = preferred_dtype
                else:
                    model_dtype = video_embeddings_pos.dtype
                mx.eval(video_embeddings_pos, video_embeddings_neg)
                _debug_stats("video_embeddings_pos", video_embeddings_pos)
                _debug_stats("video_embeddings_neg", video_embeddings_neg)

            # For distilled stage-1/stage-2 (non-dev pipelines), reuse the positive
            # embedding as the single conditioning embedding.
            if not is_dev_pipeline:
                text_embeddings = video_embeddings_pos
                audio_embeddings = audio_embeddings_pos
        else:
            # Distilled pipeline - single embedding
            if audio:
                text_embeddings, audio_embeddings = text_encoder(prompt, return_audio_embeddings=True)
                if preferred_dtype is not None:
                    text_embeddings = text_embeddings.astype(preferred_dtype)
                    audio_embeddings = audio_embeddings.astype(preferred_dtype)
                mx.eval(text_embeddings, audio_embeddings)
                _debug_stats("text_embeddings", text_embeddings)
                _debug_stats("audio_embeddings", audio_embeddings)
            else:
                text_embeddings, _ = text_encoder(prompt, return_audio_embeddings=False)
                audio_embeddings = None
                if preferred_dtype is not None:
                    text_embeddings = text_embeddings.astype(preferred_dtype)
                mx.eval(text_embeddings)
                _debug_stats("text_embeddings", text_embeddings)
            model_dtype = preferred_dtype or text_embeddings.dtype

    del text_encoder
    mx.clear_cache()
    _log_memory("text encoder freed", mem_log)

    # Load transformer (stage-1)
    transformer_desc = f"🤖 Loading {pipeline_name.lower()} transformer"
    if joint_audio:
        transformer_desc += " (A/V mode)"
    elif audio and separate_audio:
        transformer_desc += " (video-only; audio separate)"
    transformer_desc += "..."

    model_type = LTXModelType.AudioVideo if joint_audio else LTXModelType.VideoOnly
    config_kwargs = dict(
        model_type=model_type,
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
    if joint_audio:
        config_kwargs.update(
            audio_num_attention_heads=32,
            audio_attention_head_dim=64,
            audio_in_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,
            audio_out_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,
            audio_cross_attention_dim=2048,
            audio_positional_embedding_max_pos=[20],
        )
    config = LTXModelConfig(**config_kwargs)
    _debug_log(
        "config: "
        f"model_type={config.model_type} num_layers={config.num_layers} "
        f"num_attention_heads={config.num_attention_heads} "
        f"attention_head_dim={config.attention_head_dim} "
        f"cross_attention_dim={config.cross_attention_dim} "
        f"caption_channels={config.caption_channels} "
        f"rope_type={config.rope_type} double_precision_rope={config.double_precision_rope}"
    )

    def _runtime_quantize_transformer(transformer_obj: nn.Module, *, label: str) -> None:
        """Quantize selected transformer blocks in-place for lower memory / faster matmul."""

        def _predicate(path: str, module) -> bool:
            if not hasattr(module, "to_quantized"):
                return False
            if "transformer_blocks" not in path:
                return False
            w = getattr(module, "weight", None)
            if w is not None and hasattr(w, "shape") and (w.shape[0] % 64 != 0):
                # Skip layers that don't match packing constraints.
                return False

            scope = str(runtime_quantize_scope or "video_core").lower()
            if scope in {"attn1", "attn1_only", "attn1-only"}:
                if "audio_" in path or "audio_to_video" in path or "video_to_audio" in path:
                    return False
                return ".attn1" in path
            if scope in {"video_core", "video-core"}:
                if "audio_" in path or "audio_to_video" in path or "video_to_audio" in path:
                    return False
                return (".attn" in path) or (".ff" in path)
            if scope == "core":
                return (
                    (".attn" in path)
                    or (".ff" in path)
                    or ("audio_attn" in path)
                    or ("audio_ff" in path)
                    or ("audio_to_video" in path)
                    or ("video_to_audio" in path)
                )
            if scope == "all":
                return True
            # Default: video_core
            if "audio_" in path or "audio_to_video" in path or "video_to_audio" in path:
                return False
            return (".attn" in path) or (".ff" in path)

        _debug_log(
            f"runtime_quantize({label}): bits={runtime_quantize_bits} group_size={runtime_quantize_group_size} "
            f"mode={runtime_quantize_mode} scope={runtime_quantize_scope}"
        )
        nn.quantize(
            transformer_obj,
            group_size=int(runtime_quantize_group_size),
            bits=int(runtime_quantize_bits),
            mode=str(runtime_quantize_mode),
            class_predicate=_predicate,
        )
        mx.eval(transformer_obj.parameters())
        transformer_obj.eval()
        if _debug_enabled():
            q_modules = sum(1 for _, m in transformer_obj.named_modules() if hasattr(m, "scales"))
            _debug_log(f"runtime_quantized({label}) modules={q_modules}")

    def _load_transformer_with_loras(lora_list: Optional[list[tuple[str, float]]]):
        transformer_local = None
        weights_override = None
        transformer_weights = None
        runtime_lora_specs = None
        if unified_weights is not None:
            transformer_weights = {
                k[len("transformer."):]: v
                for k, v in unified_weights.items()
                if k.startswith("transformer.")
            }
            if not transformer_weights:
                transformer_weights = None
        if lora_list:
            from mlx_video.lora import LoraSpec, apply_lora_to_weights, apply_lora_to_model, has_quantized_weights
            # Avoid loading huge quantized safetensors just to detect quantization.
            # For file-based weights, scan the safetensors header for `.scales`/`.biases` keys.
            # When we plan to runtime-quantize, treat LoRAs as adapters (no merge) to avoid
            # quantizing LoRA-merged weights.
            force_adapter_mode = bool(runtime_quantize)
            is_quantized = bool(force_adapter_mode)
            if transformer_weights is not None:
                is_quantized = is_quantized or has_quantized_weights(transformer_weights)
            else:
                try:
                    import json
                    import struct

                    with open(transformer_weight_path, "rb") as f:
                        header_len = struct.unpack("<Q", f.read(8))[0]
                        header = json.loads(f.read(header_len))
                    is_quantized = is_quantized or any(
                        (k.endswith(".scales") or k.endswith(".biases"))
                        for k in header.keys()
                        if k != "__metadata__"
                    )
                except Exception:
                    # Fallback to the old behavior (may be memory-heavy).
                    is_quantized = is_quantized or has_quantized_weights(mx.load(str(transformer_weight_path)))

            lora_specs = [LoraSpec(Path(path), float(strength)) for path, strength in lora_list]

            if is_quantized:
                # Quantized base weights: apply LoRA as a runtime residual adapter.
                # This avoids dequantize/re-quantize cycles that can introduce severe artifacts ("snow").
                runtime_lora_specs = lora_specs
                if transformer_weights is not None:
                    weights_override = transformer_weights
            else:
                raw_weights = transformer_weights if transformer_weights is not None else mx.load(str(transformer_weight_path))
                weights_override = apply_lora_to_weights(raw_weights, lora_specs, verbose=verbose)
        elif transformer_weights is not None:
            weights_override = transformer_weights

        if transformer_local is None:
            transformer_local = LTXModel.from_pretrained(
                model_path=unified_path if transformer_weights is not None else transformer_weight_path,
                config=config,
                # Strict loading is important for quantized weights: missing or mismatched
                # params will silently produce "snow"/static frames when left uninitialized.
                strict=True,
                weights_override=weights_override,
            )

        if runtime_lora_specs:
            from mlx_video.lora import apply_lora_to_model

            if verbose:
                console.print("[dim]LoRA on quantized model: attaching runtime adapters (no re-quantization).[/]")
            apply_lora_to_model(transformer_local, runtime_lora_specs, verbose=verbose)
        return transformer_local

    with phase_timer.phase("stage1_transformer_load"):
        with console.status(f"[blue]{transformer_desc}[/]", spinner="dots"):
            transformer = _load_transformer_with_loras(loras)

        console.print("[green]✓[/] Transformer loaded")
    _log_memory("transformer loaded", mem_log)

    if runtime_quantize:
        with phase_timer.phase("stage1_runtime_quantize"):
            with console.status("[magenta]🧮 Runtime quantizing stage-1 transformer...[/]", spinner="dots"):
                _runtime_quantize_transformer(transformer, label="stage1")
        console.print("[green]✓[/] Stage-1 transformer quantized")
        _log_memory("stage1 transformer quantized", mem_log)

    # ==========================================================================
    # Pipeline-specific generation logic
    # ==========================================================================

    if is_distilled_pipeline:
        # ======================================================================
        # DISTILLED PIPELINE: Two-stage with upsampling
        # ======================================================================
        if stage1_steps < 1 or stage1_steps > (len(STAGE_1_SIGMAS) - 1):
            raise ValueError("--stage1-steps must be between 1 and 8.")
        if stage2_steps not in (1, 2, 3):
            raise ValueError("--stage2-steps must be 1, 2, or 3.")

        # Load VAE encoder for conditioning (images/video)
        stage1_conditionings = []
        stage2_conditionings = []
        stage1_video_conditionings = []

        if is_i2v or video_conditionings:
            with phase_timer.phase("cond_encode"):
                with console.status("[blue]🖼️  Loading VAE encoder and encoding image...[/]", spinner="dots"):
                    vae_encoder = load_vae_encoder(
                        str(_resolve_vae_source()),
                        weights_override=_unified_subset("vae_encoder."),
                    )
                    mx.eval(vae_encoder.parameters())

                    for img_path, frame_idx, strength in images_list:
                        input_image = load_image(img_path, height=height // 2, width=width // 2, dtype=model_dtype)
                        stage1_image_tensor = prepare_image_for_encoding(input_image, height // 2, width // 2, dtype=model_dtype)
                        stage1_latent = vae_encoder(stage1_image_tensor)
                        mx.eval(stage1_latent)

                        input_image = load_image(img_path, height=height, width=width, dtype=model_dtype)
                        stage2_image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
                        stage2_latent = vae_encoder(stage2_image_tensor)
                        mx.eval(stage2_latent)
                        resolved_idx = _resolve_frame_idx(frame_idx)

                        if conditioning_mode == "guide":
                            stage1_conditionings.append(
                                VideoConditionByKeyframeIndex(keyframes=stage1_latent, frame_idx=resolved_idx, strength=strength)
                            )
                            stage2_conditionings.append(
                                VideoConditionByKeyframeIndex(keyframes=stage2_latent, frame_idx=resolved_idx, strength=strength)
                            )
                        else:
                            stage1_conditionings.append(
                                VideoConditionByLatentIndex(latent=stage1_latent, frame_idx=resolved_idx, strength=strength)
                            )
                            stage2_conditionings.append(
                                VideoConditionByLatentIndex(latent=stage2_latent, frame_idx=resolved_idx, strength=strength)
                            )

                    # Video conditioning (IC-LoRA)
                    for vid_path, frame_idx, strength in video_conditionings:
                        frames = load_video(vid_path, height=height // 2, width=width // 2, frame_cap=num_frames, dtype=model_dtype)
                        video_tensor = prepare_video_for_encoding(frames, height // 2, width // 2, dtype=model_dtype)
                        video_latent = vae_encoder(video_tensor)
                        mx.eval(video_latent)
                        resolved_idx = _resolve_frame_idx(frame_idx)
                        stage1_video_conditionings.append(
                            VideoConditionByKeyframeIndex(keyframes=video_latent, frame_idx=resolved_idx, strength=strength)
                        )

                    del vae_encoder
                    mx.clear_cache()
                console.print("[green]✓[/] VAE encoder loaded and image encoded")
            if verbose:
                if stage1_conditionings:
                    console.print(f"[dim]Stage1 image conditionings: {len(stage1_conditionings)}[/]")
                    for idx, cond in enumerate(stage1_conditionings):
                        shape = tuple(cond.latent.shape) if hasattr(cond, "latent") else tuple(cond.keyframes.shape)
                        console.print(f"[dim]  - {idx}: shape={shape}, frame={cond.frame_idx}, strength={cond.strength}[/]")
                if stage1_video_conditionings:
                    console.print(f"[dim]Stage1 video conditionings: {len(stage1_video_conditionings)}[/]")
                    for idx, cond in enumerate(stage1_video_conditionings):
                        shape = tuple(cond.keyframes.shape)
                        console.print(f"[dim]  - {idx}: shape={shape}, frame={cond.frame_idx}, strength={cond.strength}[/]")

        # Stage 1
        stage1_sigmas_list = _subsample_sigmas(STAGE_1_SIGMAS, stage1_steps, sigma_subsample)

        console.print(f"\n[bold yellow]⚡ Stage 1:[/] Generating at {width//2}x{height//2} ({stage1_steps} steps)")
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = None
        audio_latents = None
        if joint_audio:
            audio_positions = create_audio_position_grid(1, audio_frames)
            audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)).astype(model_dtype)
            mx.eval(audio_positions, audio_latents)

        # Apply conditioning (images + optional video)
        state1 = None
        if stage1_conditionings or stage1_video_conditionings:
            latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
            state1 = LatentState(
                latent=mx.zeros(latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            state1 = apply_conditioning(state1, stage1_conditionings + stage1_video_conditionings)

            noise = mx.random.normal(latent_shape, dtype=model_dtype)
            noise_scale = mx.array(stage1_sigmas_list[0], dtype=model_dtype)
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal((1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype)
            mx.eval(latents)
        _debug_stats("latents_stage1_initial", latents)
        if audio_latents is not None:
            _debug_stats("audio_latents_stage1_initial", audio_latents)

        with phase_timer.phase("stage1_denoise"):
            stage1_compile_effective = bool(compile_step)
            latents, audio_latents = denoise_distilled(
                latents, positions, text_embeddings, transformer, stage1_sigmas_list,
                verbose=verbose, state=state1,
                audio_latents=audio_latents, audio_positions=audio_positions, audio_embeddings=audio_embeddings,
                eval_interval=eval_interval,
                compile_step=stage1_compile_effective,
                compile_shapeless=compile_shapeless,
                fp32_euler=fp32_euler,
                ui_phase="stage1",
            )
        _log_memory("stage1 complete", mem_log)

        # Upsample latents
        with phase_timer.phase("upsample"):
            with console.status("[magenta]🔍 Upsampling latents 2x...[/]", spinner="dots"):
                upsampler = load_upsampler(str(_resolve_upsampler_weight("ltx-2-spatial-upscaler-x2-1.0.safetensors")))
                mx.eval(upsampler.parameters())

                vae_decoder = load_vae_decoder(
                    str(_resolve_vae_source()),
                    timestep_conditioning=None,
                    weights_override=_unified_subset("vae_decoder."),
                )

                latents = upsample_latents(latents, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
                mx.eval(latents)
                _debug_stats("latents_after_upsample", latents)

                del upsampler
                mx.clear_cache()
            console.print("[green]✓[/] Latents upsampled")

        # Stage 2
        stage2_sigmas_list = _subsample_refinement_sigmas(STAGE_2_SIGMAS, stage2_steps, sigma_subsample)
        console.print(f"\n[bold yellow]⚡ Stage 2:[/] Refining at {width}x{height} ({stage2_steps} steps)")
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)
        stage2_compile_effective = bool(compile_step)

        if stage2_model_repo and distilled_loras:
            raise ValueError("--stage2-model-repo cannot be combined with --distilled-lora (stage-2 LoRA).")

        if use_stage2_dev and stage2_model_repo is None:
            # Avoid a confusing FileNotFoundError later; stage-2 dev needs dev weights.
            candidate_dev = (
                model_path / f"ltx-2-19b-dev-{bits_hint}-mlx.safetensors"
                if bits_hint
                else model_path / "ltx-2-19b-dev-mlx.safetensors"
            )
            candidate_dev_fallback = model_path / "ltx-2-19b-dev.safetensors"
            if not candidate_dev.exists() and not candidate_dev_fallback.exists():
                raise ValueError(
                    "--stage2-dev requires dev weights for stage-2 refinement. "
                    "Pass --stage2-model-repo pointing to a dev model repo/directory "
                    "(for example: ../converted/ltx2-dev-8bit-mlx)."
                )

        if stage2_model_repo or use_stage2_dev or distilled_loras:
            with phase_timer.phase("stage2_transformer_load"):
                del transformer
                mx.clear_cache()
                _log_memory("before stage2 transformer load", mem_log)

                if distilled_loras:
                    with console.status("[blue]🤖 Loading stage-2 transformer (distilled LoRA)...[/]", spinner="dots"):
                        transformer = _load_transformer_with_loras(distilled_loras)
                    console.print("[green]✓[/] Stage-2 transformer loaded")
                else:
                    stage2_path = model_path if stage2_model_repo is None else get_model_path(stage2_model_repo)
                    stage2_is_dev = bool(use_stage2_dev)
                    stage2_kind = "dev" if stage2_is_dev else "distilled"

                    stage2_repo_hint = f"{stage2_model_repo or model_repo}".lower()
                    stage2_bits_hint: str | None = None
                    if any(x in stage2_repo_hint for x in ("8bit", "q8", "int8")):
                        stage2_bits_hint = "8bit"
                    elif any(x in stage2_repo_hint for x in ("4bit", "q4", "int4")):
                        stage2_bits_hint = "4bit"
                    else:
                        stage2_bits_hint = bits_hint

                    stage2_candidates: list[Path] = []
                    if stage2_bits_hint:
                        stage2_candidates.append(
                            stage2_path / f"ltx-2-19b-{stage2_kind}-{stage2_bits_hint}-mlx.safetensors"
                        )
                    stage2_candidates.append(stage2_path / f"ltx-2-19b-{stage2_kind}-mlx.safetensors")
                    stage2_candidates.append(stage2_path / f"ltx-2-19b-{stage2_kind}.safetensors")
                    stage2_weights_path = next((p for p in stage2_candidates if p.exists()), stage2_candidates[-1])
                    if not stage2_weights_path.exists():
                        raise FileNotFoundError(
                            f"Stage-2 weights not found at {stage2_weights_path}. "
                            "Ensure the stage-2 model repo contains LTX-2 weights."
                        )
                    stage2_desc = f"{stage2_kind} transformer" + (f" from {stage2_path}" if stage2_model_repo else "")
                    with console.status(f"[blue]🤖 Loading stage-2 {stage2_desc}...[/]", spinner="dots"):
                        transformer = LTXModel.from_pretrained(
                            model_path=stage2_weights_path,
                            config=config,
                            # Stage-2 is the same architecture class as stage-1; load strictly to
                            # avoid silent partial loads that manifest as static artifacts.
                            strict=True,
                        )
                    console.print("[green]✓[/] Stage-2 transformer loaded")
                    if runtime_quantize:
                        with console.status("[magenta]🧮 Runtime quantizing stage-2 transformer...[/]", spinner="dots"):
                            _runtime_quantize_transformer(transformer, label="stage2")
                        console.print("[green]✓[/] Stage-2 transformer quantized")

        state2 = None
        if stage2_conditionings and verbose:
            console.print(f"[dim]Stage2 image conditionings: {len(stage2_conditionings)}[/]")
            for idx, cond in enumerate(stage2_conditionings):
                shape = tuple(cond.latent.shape) if hasattr(cond, "latent") else tuple(cond.keyframes.shape)
                console.print(f"[dim]  - {idx}: shape={shape}, frame={cond.frame_idx}, strength={cond.strength}[/]")
        if stage2_conditionings:
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            # For IC-LoRA, stage2 uses replace for images (as in PyTorch)
            state2 = apply_conditioning(state2, stage2_conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(stage2_sigmas_list[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)

            if joint_audio and audio_latents is not None:
                audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
                one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
                audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
                mx.eval(audio_latents)
        else:
            noise_scale = mx.array(stage2_sigmas_list[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - stage2_sigmas_list[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

            if joint_audio and audio_latents is not None:
                audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
                audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
                mx.eval(audio_latents)
        _debug_stats("latents_stage2_initial", latents)
        if audio_latents is not None:
            _debug_stats("audio_latents_stage2_initial", audio_latents)

        with phase_timer.phase("stage2_denoise"):
            if use_stage2_dev:
                # Use dev denoiser (CFG) for stage-2 refinement while keeping the
                # distilled two-stage structure for speed. We keep the distilled stage-2
                # sigma schedule and just swap the denoiser to CFG.
                sigmas_stage2 = mx.array(stage2_sigmas_list, dtype=model_dtype)
                mx.eval(sigmas_stage2)
                if joint_audio:
                    latents, audio_latents = denoise_dev_av(
                        latents, audio_latents,
                        positions, audio_positions,
                        video_embeddings_pos, video_embeddings_neg,
                        audio_embeddings_pos, audio_embeddings_neg,
                        transformer, sigmas_stage2, cfg_scale=cfg_scale, verbose=verbose, video_state=state2,
                        eval_interval=eval_interval,
                        compile_step=stage2_compile_effective,
                        compile_shapeless=compile_shapeless,
                        cfg_batch=cfg_batch,
                        ui_phase="stage2",
                    )
                else:
                    latents = denoise_dev(
                        latents, positions, video_embeddings_pos, video_embeddings_neg,
                        transformer, sigmas_stage2, cfg_scale=cfg_scale, verbose=verbose, state=state2,
                        eval_interval=eval_interval,
                        compile_step=stage2_compile_effective,
                        compile_shapeless=compile_shapeless,
                        cfg_batch=cfg_batch,
                        ui_phase="stage2",
                    )
            else:
                latents, audio_latents = denoise_distilled(
                    latents, positions, text_embeddings, transformer, stage2_sigmas_list,
                    verbose=verbose, state=state2,
                    audio_latents=audio_latents, audio_positions=audio_positions, audio_embeddings=audio_embeddings,
                    eval_interval=eval_interval,
                    compile_step=stage2_compile_effective,
                    compile_shapeless=compile_shapeless,
                    fp32_euler=fp32_euler,
                    ui_phase="stage2",
                )
        _log_memory("stage2 complete", mem_log)

    else:
        # ======================================================================
        # DEV PIPELINE: Single-stage with CFG
        # ======================================================================

        # Load VAE encoder for I2V
        dev_conditionings = []
        if is_i2v:
            with console.status("[blue]🖼️  Loading VAE encoder and encoding image...[/]", spinner="dots"):
                vae_encoder = load_vae_encoder(
                    str(_resolve_vae_source()),
                    weights_override=_unified_subset("vae_encoder."),
                )
                mx.eval(vae_encoder.parameters())

                for img_path, frame_idx, strength in images_list:
                    input_image = load_image(img_path, height=height, width=width, dtype=model_dtype)
                    image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
                    image_latent = vae_encoder(image_tensor)
                    mx.eval(image_latent)
                    resolved_idx = _resolve_frame_idx(frame_idx)

                    if conditioning_mode == "guide":
                        dev_conditionings.append(
                            VideoConditionByKeyframeIndex(keyframes=image_latent, frame_idx=resolved_idx, strength=strength)
                        )
                    else:
                        dev_conditionings.append(
                            VideoConditionByLatentIndex(latent=image_latent, frame_idx=resolved_idx, strength=strength)
                        )

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image encoded")

        # Generate sigma schedule
        num_tokens = latent_frames * latent_h * latent_w
        sigmas = ltx2_scheduler(steps=num_inference_steps, num_tokens=num_tokens)
        mx.eval(sigmas)
        console.print(f"[dim]Sigma schedule: {sigmas[0].item():.4f} → {sigmas[-2].item():.4f} → {sigmas[-1].item():.4f}[/]")

        gen_dims_str = f"{output_width}x{output_height}"
        if crop_params is not None:
            gen_dims_str += f" (internal {width}x{height})"
        console.print(f"\n[bold yellow]⚡ Generating:[/] {gen_dims_str} ({num_inference_steps} steps, CFG={cfg_scale})")
        mx.random.seed(seed)

        video_positions = create_position_grid(1, latent_frames, latent_h, latent_w)
        mx.eval(video_positions)

        audio_positions = None
        audio_latents = None
        if joint_audio:
            audio_positions = create_audio_position_grid(1, audio_frames)
            audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS), dtype=model_dtype)
            mx.eval(audio_positions, audio_latents)

        # Initialize latents with optional I2V conditioning
        video_state = None
        video_latent_shape = (1, 128, latent_frames, latent_h, latent_w)
        if dev_conditionings:
            video_state = LatentState(
                latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            video_state = apply_conditioning(video_state, dev_conditionings)

            noise = mx.random.normal(video_latent_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = video_state.denoise_mask * noise_scale
            video_state = LatentState(
                latent=noise * scaled_mask + video_state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=video_state.clean_latent,
                denoise_mask=video_state.denoise_mask,
            )
            latents = video_state.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(video_latent_shape, dtype=model_dtype)
            mx.eval(latents)
        _debug_stats("latents_initial", latents)
        if audio_latents is not None:
            _debug_stats("audio_latents_initial", audio_latents)

        # Denoise with CFG
        if metal_capture and capture_phase in ("denoise", "all") and capture_path is not None:
            try:
                capture_started = _start_metal_capture(capture_path)
                if capture_started:
                    console.print(f"[dim]Metal capture started: {capture_path}[/]")
            except Exception as exc:
                console.print(f"[yellow]⚠️  Could not start Metal capture: {exc}[/]")
        with phase_timer.phase("dev_denoise"):
            if joint_audio:
                latents, audio_latents = denoise_dev_av(
                    latents, audio_latents,
                    video_positions, audio_positions,
                    video_embeddings_pos, video_embeddings_neg,
                    audio_embeddings_pos, audio_embeddings_neg,
                    transformer, sigmas, cfg_scale=cfg_scale, verbose=verbose, video_state=video_state,
                    eval_interval=eval_interval,
                    compile_step=compile_step,
                    compile_shapeless=compile_shapeless,
                    cfg_batch=cfg_batch,
                    ui_phase="dev",
                )
            else:
                latents = denoise_dev(
                    latents, video_positions, video_embeddings_pos, video_embeddings_neg,
                    transformer, sigmas, cfg_scale=cfg_scale, verbose=verbose, state=video_state,
                    eval_interval=eval_interval,
                    compile_step=compile_step,
                    compile_shapeless=compile_shapeless,
                    cfg_batch=cfg_batch,
                    ui_phase="dev",
                )
        if capture_started and capture_phase in ("denoise", "all"):
            mx.metal.stop_capture()
            capture_started = False
            console.print(f"[dim]Metal capture saved: {capture_path}[/]")
        _log_memory("denoise complete", mem_log)

        # Load VAE decoder (for dev pipeline, loaded here instead of during upsampling)
        vae_decoder = load_vae_decoder(
            str(_resolve_vae_source()),
            timestep_conditioning=None,
            weights_override=_unified_subset("vae_decoder."),
        )

    del transformer
    mx.clear_cache()
    _log_memory("after transformer free", mem_log)

    # ==========================================================================
    # Decode and save outputs (common to both pipelines)
    # ==========================================================================

    if metal_capture and capture_phase in ("decode", "all") and capture_path is not None:
        try:
            capture_started = _start_metal_capture(capture_path)
            if capture_started:
                console.print(f"[dim]Metal capture started: {capture_path}[/]")
        except Exception as exc:
            console.print(f"[yellow]⚠️  Could not start Metal capture: {exc}[/]")

    console.print("\n[blue]🎞️  Decoding video...[/]")

    # Select tiling configuration
    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        console.print(f"[yellow]  Unknown tiling mode '{tiling}', using auto[/]")
        tiling_config = TilingConfig.auto(height, width, num_frames)

    # Stream mode needs incremental decode so we can write the temp MP4 and emit
    # live previews/progress. `TilingConfig.auto()` returns `None` for smaller
    # videos, which would silently disable stream mode entirely.
    if stream and tiling_config is None:
        try:
            # Choose a conservative temporal tile so overhead stays low while
            # still producing incremental callbacks for the stream writer.
            #
            # Requirements: tile_size >= 16 and divisible by 8.
            tile_size = 64
            if num_frames < tile_size:
                tile_size = max(16, (num_frames // 8) * 8)
                if tile_size <= 0:
                    tile_size = 16
            overlap = 24 if tile_size >= 64 else 8
            tiling_config = TilingConfig.temporal_only(tile_size=tile_size, overlap=overlap)
            console.print(
                f"[dim]  Stream enabled: forcing temporal tiling ({tile_size}f/{overlap}f) for incremental decode[/]"
            )
        except Exception:
            # Fallback to default temporal tiling if something goes wrong.
            tiling_config = TilingConfig.temporal_only()
            console.print("[dim]  Stream enabled: forcing temporal tiling for incremental decode[/]")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_np: np.ndarray | None = None

    # Stream mode (low memory): stream decoded frames into a video encoder as they become available.
    stream_progress = None
    stream_task = None
    stream_video_path: Path | None = None
    stream_cv2 = None
    stream_ffmpeg = None
    writer_height, writer_width = output_height, output_width
    if crop_params is None:
        writer_height, writer_width = height, width

    if stream and tiling_config is not None:
        stream_output_path = output_path.with_suffix('.temp.mp4') if audio else output_path
        stream_video_path = stream_output_path

        # Prefer ffmpeg pipe encoding when requested (better quality control, avoids cv2 artifacts).
        if video_encoder == "ffmpeg":
            try:
                import subprocess

                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is None:
                    raise FileNotFoundError("ffmpeg not found")
                stream_crf = int(os.environ.get("LTX_STREAM_CRF", "18"))
                stream_preset = os.environ.get("LTX_STREAM_PRESET", "veryfast")
                stream_codec = os.environ.get("LTX_STREAM_CODEC", "libx264")

                cmd = [
                    ffmpeg,
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{writer_width}x{writer_height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    stream_codec,
                    "-preset",
                    stream_preset,
                    "-crf",
                    str(stream_crf),
                    "-pix_fmt",
                    "yuv420p",
                    str(stream_output_path),
                ]
                stream_ffmpeg = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if stream_ffmpeg.stdin is None:
                    raise RuntimeError("ffmpeg stdin not available")
            except Exception as exc:
                console.print(f"[yellow]⚠️  ffmpeg stream writer failed; falling back to OpenCV. ({exc})[/]")
                stream_ffmpeg = None

        # OpenCV fallback (kept for environments without ffmpeg).
        if stream_ffmpeg is None:
            import cv2

            for codec in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(stream_output_path), fourcc, fps, (writer_width, writer_height))
                if out.isOpened():
                    stream_cv2 = out
                    break
                out.release()
            if stream_cv2 is None:
                console.print("[yellow]⚠️  Stream writer failed to open; falling back to non-stream write[/]")
                stream_video_path = None

        if stream_video_path is not None and (stream_cv2 is not None or stream_ffmpeg is not None):
            stream_progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )
            stream_progress.start()
            stream_task = stream_progress.add_task("[cyan]Streaming frames[/]", total=num_frames)

        # Optional live preview image written during streaming decode.
        # The temp MP4 may not be playable until finalized (moov atom), so we emit a
        # periodically-updated JPEG that the UI can poll while generation is running.
        preview_path_env = os.environ.get("MLX_VIDEO_PREVIEW_PATH")
        preview_path = Path(preview_path_env).expanduser() if preview_path_env else None
        preview_every = int(os.environ.get("MLX_VIDEO_PREVIEW_EVERY", "12"))
        preview_max_dim = int(os.environ.get("MLX_VIDEO_PREVIEW_MAX_DIM", "512"))
        preview_quality = int(os.environ.get("MLX_VIDEO_PREVIEW_QUALITY", "85"))
        last_preview_idx = -1
        progress_echo = os.environ.get("MLX_VIDEO_PROGRESS_ECHO") == "1"
        try:
            progress_echo_every = int(
                os.environ.get(
                    "MLX_VIDEO_DECODE_ECHO_EVERY",
                    os.environ.get("MLX_VIDEO_PROGRESS_ECHO_EVERY", str(preview_every)),
                )
            )
        except Exception:
            progress_echo_every = preview_every
        last_progress_idx = -1
        t_decode0 = time.perf_counter()
        preview_pil = None
        if preview_path is not None:
            try:
                from PIL import Image as _PILImage
                preview_pil = _PILImage
                preview_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                preview_path = None

        def on_frames_ready(frames: mx.array, _start_idx: int):
            nonlocal last_preview_idx, last_progress_idx
            frames = mx.squeeze(frames, axis=0)
            frames = mx.transpose(frames, (1, 2, 3, 0))
            frames = mx.clip((frames + 1.0) / 2.0, 0.0, 1.0)
            frames = (frames * 255).astype(mx.uint8)
            frames_np = np.array(frames)

            if crop_params is not None:
                top, left, out_h, out_w = crop_params
                frames_np = frames_np[:, top : top + out_h, left : left + out_w, :]

            # NOTE: `frames_np` is RGB uint8 (HWC). OpenCV expects BGR while ffmpeg rawvideo
            # expects the RGB bytes as-is ("rgb24").
            if stream_ffmpeg is not None and stream_ffmpeg.stdin is not None:
                for i, frame in enumerate(frames_np):
                    try:
                        stream_ffmpeg.stdin.write(frame.tobytes(order="C"))
                    except Exception:
                        # If ffmpeg dies mid-stream, stop writing frames. We'll fall back later.
                        break

                    if stream_progress is not None and stream_task is not None:
                        stream_progress.advance(stream_task)
                    if progress_echo and progress_echo_every > 0:
                        idx = _start_idx + i
                        if idx == 0 or idx == num_frames - 1 or (idx - last_progress_idx) >= progress_echo_every:
                            try:
                                done = idx + 1
                                elapsed = time.perf_counter() - t_decode0
                                eta_s = (elapsed / max(1, done)) * max(0, num_frames - done)
                                print(f"Streaming frames {done}/{num_frames} ETA {_format_eta(eta_s)}", flush=True)
                                last_progress_idx = idx
                            except Exception:
                                pass
                    if preview_path is not None and preview_pil is not None and preview_every > 0:
                        idx = _start_idx + i
                        if idx == 0 or (idx - last_preview_idx) >= preview_every:
                            try:
                                img = preview_pil.fromarray(frame)
                                if preview_max_dim > 0:
                                    img.thumbnail((preview_max_dim, preview_max_dim), resample=preview_pil.BILINEAR)
                                tmp = preview_path.with_suffix(preview_path.suffix + ".tmp")
                                img.save(str(tmp), format="JPEG", quality=preview_quality, optimize=True)
                                os.replace(str(tmp), str(preview_path))
                                last_preview_idx = idx
                                _ui_event(
                                    {
                                        "kind": "progress",
                                        "phase": "decode",
                                        "current": int(idx + 1),
                                        "total": int(num_frames),
                                        "percent": 100.0 * float(idx + 1) / float(max(1, num_frames)),
                                    }
                                )
                            except Exception:
                                pass
            elif stream_cv2 is not None:
                for i, frame in enumerate(frames_np):
                    stream_cv2.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if stream_progress is not None and stream_task is not None:
                        stream_progress.advance(stream_task)
                    if progress_echo and progress_echo_every > 0:
                        idx = _start_idx + i
                        if idx == 0 or idx == num_frames - 1 or (idx - last_progress_idx) >= progress_echo_every:
                            try:
                                done = idx + 1
                                elapsed = time.perf_counter() - t_decode0
                                eta_s = (elapsed / max(1, done)) * max(0, num_frames - done)
                                print(f"Streaming frames {done}/{num_frames} ETA {_format_eta(eta_s)}", flush=True)
                                last_progress_idx = idx
                            except Exception:
                                pass
                    if preview_path is not None and preview_pil is not None and preview_every > 0:
                        idx = _start_idx + i
                        if idx == 0 or (idx - last_preview_idx) >= preview_every:
                            try:
                                img = preview_pil.fromarray(frame)
                                if preview_max_dim > 0:
                                    img.thumbnail((preview_max_dim, preview_max_dim), resample=preview_pil.BILINEAR)
                                tmp = preview_path.with_suffix(preview_path.suffix + ".tmp")
                                img.save(str(tmp), format="JPEG", quality=preview_quality, optimize=True)
                                os.replace(str(tmp), str(preview_path))
                                last_preview_idx = idx
                                _ui_event(
                                    {
                                        "kind": "progress",
                                        "phase": "decode",
                                        "current": int(idx + 1),
                                        "total": int(num_frames),
                                        "percent": 100.0 * float(idx + 1) / float(max(1, num_frames)),
                                    }
                                )
                            except Exception:
                                pass
    else:
        on_frames_ready = None

    if tiling_config is not None:
        spatial_info = f"{tiling_config.spatial_config.tile_size_in_pixels}px" if tiling_config.spatial_config else "none"
        temporal_info = f"{tiling_config.temporal_config.tile_size_in_frames}f" if tiling_config.temporal_config else "none"
        console.print(f"[dim]  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}[/]")
        _debug_stats("latents_pre_decode", latents)
    else:
        console.print("[dim]  Tiling: disabled[/]")
        _debug_stats("latents_pre_decode", latents)

    with phase_timer.phase("vae_decode"):
        if tiling_config is not None:
            # Auto tiling is conservative for many M-chip configs; try a fast non-tiled
            # decode first and fall back to tiling on Metal OOM/resource errors.
            if tiling == "auto" and not stream:
                try:
                    console.print("[dim]  Auto tiling: trying non-tiled decode (fast path)[/]")
                    video = vae_decoder(latents, chunked_conv=False)
                except Exception as exc:
                    if not _looks_like_metal_oom(exc):
                        raise
                    console.print("[yellow]⚠️  Non-tiled decode failed; retrying with chunked conv[/]")
                    try:
                        video = vae_decoder(latents, chunked_conv=True)
                    except Exception as exc2:
                        if not _looks_like_metal_oom(exc2):
                            raise
                        console.print("[yellow]⚠️  Chunked decode failed; falling back to tiled decode[/]")
                        video = vae_decoder.decode_tiled(
                            latents,
                            tiling_config=tiling_config,
                            tiling_mode=tiling,
                            debug=verbose,
                            on_frames_ready=on_frames_ready,
                        )
            else:
                video = vae_decoder.decode_tiled(
                    latents,
                    tiling_config=tiling_config,
                    tiling_mode=tiling,
                    debug=verbose,
                    on_frames_ready=on_frames_ready,
                )
        else:
            video = vae_decoder(latents, chunked_conv=False)
        mx.eval(video)
        mx.clear_cache()

    # Close stream writer
    if stream_video_path is not None and (stream_cv2 is not None or stream_ffmpeg is not None):
        stream_ok = True
        if stream_cv2 is not None:
            try:
                stream_cv2.release()
            except Exception:
                stream_ok = False
        if stream_ffmpeg is not None:
            try:
                if stream_ffmpeg.stdin is not None:
                    stream_ffmpeg.stdin.close()
                rc = stream_ffmpeg.wait()
                if rc != 0:
                    stream_ok = False
                    err = b""
                    if stream_ffmpeg.stderr is not None:
                        err = stream_ffmpeg.stderr.read()
                    console.print(
                        f"[yellow]⚠️  ffmpeg stream encode failed (rc={rc}):[/]\n{err.decode(errors='ignore')}"
                    )
            except Exception as exc:
                stream_ok = False
                console.print(f"[yellow]⚠️  ffmpeg stream encode failed: {exc}[/]")

        if stream_progress is not None:
            stream_progress.stop()
        final_stream_path = output_path.with_suffix('.temp.mp4') if audio else output_path
        # If the stream output didn't land where expected, fall back to output_path.
        if not final_stream_path.exists() and output_path.exists():
            final_stream_path = output_path
        stream_video_path = final_stream_path

        stream_output_ok = (
            stream_ok
            and final_stream_path.exists()
            and final_stream_path.stat().st_size > 0
        )

        # Only materialize full `video_np` if we need it (fallback encode or save_frames).
        if (not stream_output_ok) or save_frames:
            with phase_timer.phase("to_uint8_numpy"):
                video = mx.squeeze(video, axis=0)
                video = mx.transpose(video, (1, 2, 3, 0))
                video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
                video = (video * 255).astype(mx.uint8)
                video_np = np.array(video)
                if crop_params is not None:
                    top, left, out_h, out_w = crop_params
                    video_np = video_np[:, top : top + out_h, left : left + out_w, :]

        if not stream_output_ok:
            console.print(f"[yellow]⚠️  Stream output missing/invalid; encoding video to {final_stream_path}[/]")
            if video_np is None:
                raise RuntimeError("Stream output invalid and video frames unavailable for fallback encoding")
            with phase_timer.phase("video_write"):
                _write_video(video_np, final_stream_path, fps, console=console, encoder=video_encoder)
            stream_video_path = final_stream_path

        console.print(f"[green]✅ Streamed video to[/] {stream_video_path}")
    else:
        with phase_timer.phase("to_uint8_numpy"):
            video = mx.squeeze(video, axis=0)
            video = mx.transpose(video, (1, 2, 3, 0))
            video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
            video = (video * 255).astype(mx.uint8)
            video_np = np.array(video)
            if crop_params is not None:
                top, left, out_h, out_w = crop_params
                video_np = video_np[:, top : top + out_h, left : left + out_w, :]

        if audio:
            temp_video_path = output_path.with_suffix('.temp.mp4')
            save_path = temp_video_path
        else:
            save_path = output_path

        try:
            with phase_timer.phase("video_write"):
                _write_video(video_np, save_path, fps, console=console, encoder=video_encoder)
            if not audio:
                console.print(f"[green]✅ Saved video to[/] {output_path}")
        except Exception as e:
            console.print(f"[red]❌ Could not save video: {e}[/]")
        if audio:
            stream_video_path = save_path

    # Free decoded video tensor as early as possible. At this point:
    # - the temp/final MP4 has been written (streamed or non-streamed)
    # - any optional `video_np` needed for fallbacks or save_frames has been materialized
    try:
        del video
    except Exception:
        pass
    mx.clear_cache()

    # Decode and save audio if enabled
    audio_np = None
    if audio:
        # If audio was requested but not generated jointly, generate it now via
        # an AudioOnly pass (typically with dev weights) and then mux into MP4.
        if audio_latents is None and separate_audio:
            if audio_steps < 1 or audio_steps > (len(STAGE_1_SIGMAS) - 1):
                raise ValueError("--audio-steps must be between 1 and 8.")
            if audio_frames is None:
                raise RuntimeError("audio_frames was not computed for separate audio generation")

            audio_ctx = audio_embeddings
            if audio_ctx is None and audio_embeddings_pos is not None:
                audio_ctx = audio_embeddings_pos
            if audio_ctx is None:
                raise RuntimeError("Audio embeddings are missing; cannot generate separate audio.")

            inferred_audio_repo = audio_model_repo
            if inferred_audio_repo is None:
                inferred_audio_repo = model_repo
                # If `model_repo` is a local directory (common in Pinokio / HF cache snapshots),
                # do NOT try to "infer" another repo by string replacing paths. That can easily
                # fabricate non-existent snapshot paths and crash HF validation.
                #
                # Only apply the distilled->dev inference when `model_repo` looks like a HF repo id.
                try:
                    _mr_path = Path(str(model_repo)).expanduser()
                    _is_local_repo = _mr_path.exists() and _mr_path.is_dir()
                except Exception:
                    _is_local_repo = False

                if (not _is_local_repo) and "distilled" in str(model_repo).lower() and "dev" not in str(model_repo).lower():
                    inferred_audio_repo = str(model_repo).replace("distilled", "dev")

                # If we're doing runtime quantization (to avoid broken / bloated pre-quant
                # snapshots), do NOT infer audio weights from the AITRADER repos. Those
                # snapshots can be extremely large and may include duplicate variants.
                # Use the known-good Dev BF16 checkpoint and quantize it at runtime.
                if runtime_quantize:
                    inferred_audio_repo = "mlx-community/LTX-2-dev-bf16"

            audio_model_path = get_model_path(inferred_audio_repo, require_files=False)
            audio_weight_candidates = [
                audio_model_path / "ltx-2-19b-dev-mlx.safetensors",
                audio_model_path / "ltx-2-19b-dev.safetensors",
                audio_model_path / "ltx-2-19b-distilled-mlx.safetensors",
                audio_model_path / "ltx-2-19b-distilled.safetensors",
                model_path / "ltx-2-19b-dev-mlx.safetensors",
                model_path / "ltx-2-19b-dev.safetensors",
                model_path / "ltx-2-19b-distilled-mlx.safetensors",
                model_path / "ltx-2-19b-distilled.safetensors",
            ]
            audio_weights_path = next((p for p in audio_weight_candidates if p.exists()), None)
            if audio_weights_path is None:
                raise FileNotFoundError("Could not find transformer weights for separate audio generation.")

            with phase_timer.phase("audio_generate"):
                console.print(f"\n[bold yellow]🎵 Audio:[/] Generating separately ({audio_steps} steps)")
                with console.status("[blue]🎵 Loading audio transformer...[/]", spinner="dots"):
                    audio_config = LTXModelConfig(
                        model_type=LTXModelType.AudioOnly,
                        # Keep video config defaults (unused) for parity with saved weights.
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
                        # Audio config (used)
                        audio_num_attention_heads=32,
                        audio_attention_head_dim=64,
                        audio_in_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,
                        audio_out_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,
                        audio_cross_attention_dim=2048,
                        audio_positional_embedding_max_pos=[20],
                    )
                    audio_transformer = LTXModel.from_pretrained(
                        model_path=audio_weights_path,
                        config=audio_config,
                        strict=False,
                    )
                console.print("[green]✓[/] Audio transformer loaded")

                # Keep audio generation consistent with the runtime-quantized video path:
                # avoid downloading or relying on pre-quant AITRADER snapshots.
                if runtime_quantize:
                    _runtime_quantize_transformer(audio_transformer, label="audio")

                # Reset seed so audio is reproducible regardless of video sampling.
                mx.random.seed(seed)
                audio_positions_sep = create_audio_position_grid(1, audio_frames)
                audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)).astype(model_dtype)
                mx.eval(audio_positions_sep, audio_latents)

                audio_sigmas_list = _subsample_sigmas(STAGE_1_SIGMAS, audio_steps, sigma_subsample)
                # Avoid compilation overhead for the short audio loop.
                audio_latents = denoise_audio_only(
                    audio_latents,
                    audio_positions_sep,
                    audio_ctx,
                    audio_transformer,
                    audio_sigmas_list,
                    verbose=verbose,
                    eval_interval=eval_interval,
                    compile_step=False,
                    compile_shapeless=False,
                    fp32_euler=fp32_euler,
                    ui_phase="audio",
                )
                del audio_transformer
                mx.clear_cache()

        if audio_latents is not None:
            with phase_timer.phase("audio_decode"):
                with console.status("[blue]🔊 Decoding audio...[/]", spinner="dots"):
                    audio_decoder = load_audio_decoder(model_path, pipeline, unified_weights=unified_weights)
                    vocoder = load_vocoder(model_path, pipeline, unified_weights=unified_weights)
                    mx.eval(audio_decoder.parameters(), vocoder.parameters())

                    mel_spectrogram = audio_decoder(audio_latents)
                    audio_waveform = vocoder(mel_spectrogram)
                    mx.eval(audio_waveform)

                    audio_np = np.array(audio_waveform.astype(mx.float32))
                    if audio_np.ndim == 3:
                        audio_np = audio_np[0]

                    del audio_decoder, vocoder
                    mx.clear_cache()
                console.print("[green]✓[/] Audio decoded")

            audio_path = Path(output_audio_path) if output_audio_path else output_path.with_suffix('.wav')
            with phase_timer.phase("audio_save"):
                save_audio(audio_np, audio_path, AUDIO_SAMPLE_RATE)
            console.print(f"[green]✅ Saved audio to[/] {audio_path}")

            with phase_timer.phase("audio_mux"):
                with console.status("[blue]🎬 Combining video and audio...[/]", spinner="dots"):
                    temp_video_path = stream_video_path or output_path.with_suffix('.temp.mp4')
                    if (not temp_video_path.exists()) or (temp_video_path.exists() and temp_video_path.stat().st_size == 0):
                        console.print(f"[yellow]⚠️  Temp video missing; re-encoding to {temp_video_path}[/]")
                        with phase_timer.phase("video_write"):
                            _write_video(video_np, temp_video_path, fps, console=console, encoder=video_encoder)
                    success = mux_video_audio(temp_video_path, audio_path, output_path)
            if success:
                console.print(f"[green]✅ Saved video with audio to[/] {output_path}")
                if temp_video_path.exists() and temp_video_path != output_path:
                    temp_video_path.unlink()
            else:
                if temp_video_path.exists():
                    temp_video_path.rename(output_path)
                    console.print(f"[yellow]⚠️  Saved video without audio to[/] {output_path}")
                else:
                    console.print("[yellow]⚠️  Audio mux failed and temp video missing; leaving output unchanged.[/]")
        else:
            console.print("[yellow]⚠️  Audio was enabled but no audio latents were produced; saving video without audio.[/]")

    del vae_decoder
    if capture_started and capture_phase in ("decode", "all"):
        mx.metal.stop_capture()
        capture_started = False
        console.print(f"[dim]Metal capture saved: {capture_path}[/]")
    mx.clear_cache()
    _log_memory("after decode", mem_log)

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        console.print(f"[green]✅ Saved {len(video_np)} frames to {frames_dir}[/]")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    console.print(Panel(
        f"[bold green]🎉 Done![/] Generated in {time_str} ({elapsed/num_frames:.2f}s/frame)\n"
        f"[bold green]✨ Peak memory:[/] {mx.get_peak_memory() / (1024 ** 3):.2f}GB",
        expand=False
    ))
    if profile:
        report = phase_timer.render(elapsed)
        if report:
            console.print(Panel(report, title="[bold]Timings[/]", expand=False))
    if profile_json_path and phase_timer.times_s:
        try:
            import json

            out_path = Path(profile_json_path)
            if profile_json_path == "auto":
                out_path = output_path.with_suffix(".profile.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "elapsed_s": float(elapsed),
                "num_frames": int(num_frames),
                "fps": float(fps),
                "pipeline": pipeline.value,
                "stage1_steps": int(stage1_steps) if is_distilled_pipeline else None,
                "stage2_steps": int(stage2_steps) if is_distilled_pipeline else None,
                "eval_interval": int(eval_interval),
                "compile_step": bool(compile_step),
                "compile_step_stage1": stage1_compile_effective,
                "compile_step_stage2": stage2_compile_effective,
                "compile_shapeless": bool(compile_shapeless),
                "fp32_euler": bool(fp32_euler),
                "audio": bool(audio),
                "audio_mode": str(audio_mode),
                "audio_steps": int(audio_steps) if audio else None,
                "phases_s": {k: float(v) for k, v in phase_timer.times_s.items()},
                "peak_memory_gb": float(mx.get_peak_memory() / (1024 ** 3)),
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"[dim]Wrote profile JSON:[/] {out_path}")
        except Exception as exc:
            console.print(f"[yellow]⚠️  Failed to write profile JSON ({exc})[/]")

    if clear_cache:
        mx.clear_cache()
        _log_memory("after clear-cache", mem_log)

    if audio:
        return video_np, audio_np
    return video_np


def main():
    class ImageConditionAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) not in (1, 3):
                msg = f"{option_string} accepts 1 or 3 args (PATH [FRAME_IDX STRENGTH]), got {len(values)}"
                raise argparse.ArgumentError(self, msg)
            path = values[0]
            if len(values) == 3:
                frame_idx = int(values[1])
                strength = float(values[2])
            else:
                frame_idx = None
                strength = None
            current = getattr(namespace, self.dest) or []
            current.append((path, frame_idx, strength))
            setattr(namespace, self.dest, current)

    class VideoConditionAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) not in (2, 3):
                msg = f"{option_string} accepts PATH STRENGTH or PATH FRAME_IDX STRENGTH"
                raise argparse.ArgumentError(self, msg)
            if len(values) == 2:
                path, strength = values[0], float(values[1])
                frame_idx = 0
            else:
                path = values[0]
                frame_idx = int(values[1])
                strength = float(values[2])
            current = getattr(namespace, self.dest) or []
            current.append((path, frame_idx, strength))
            setattr(namespace, self.dest, current)

    class LoraAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) not in (1, 2):
                msg = f"{option_string} accepts PATH or PATH STRENGTH, got {len(values)}"
                raise argparse.ArgumentError(self, msg)
            path = values[0]
            strength = float(values[1]) if len(values) == 2 else 1.0
            current = getattr(namespace, self.dest) or []
            current.append((path, strength))
            setattr(namespace, self.dest, current)

    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2 (Distilled or Dev pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distilled pipeline (two-stage, fast, no CFG)
  python -m mlx_video.generate --prompt "A cat walking on grass"
  python -m mlx_video.generate --prompt "Ocean waves" --pipeline distilled

  # Dev pipeline (single-stage, CFG, higher quality)
  python -m mlx_video.generate --prompt "A cat walking" --pipeline dev --cfg-scale 4.0
  python -m mlx_video.generate --prompt "Ocean waves" --pipeline dev --steps 50

  # Image-to-Video (works with both pipelines)
  python -m mlx_video.generate --prompt "A person dancing" --image photo.jpg
  python -m mlx_video.generate --prompt "Waves crashing" --image beach.png --pipeline dev

  # With Audio (works with both pipelines)
  python -m mlx_video.generate --prompt "Ocean waves crashing" --audio
  python -m mlx_video.generate --prompt "A jazz band playing" --audio --pipeline dev
        """
    )

    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text description of the video to generate")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="distilled",
        choices=["distilled", "dev", "keyframe", "ic_lora"],
        help="Pipeline type: distilled (two-stage), dev (single-stage CFG), keyframe (two-stage keyframe), ic_lora (two-stage with video conditioning)",
    )
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt for CFG (dev pipeline only)")
    parser.add_argument("--height", "-H", type=int, default=512, help="Output video height")
    parser.add_argument("--width", "-W", type=int, default=512, help="Output video width")
    parser.add_argument("--num-frames", "-n", type=int, default=33, help="Number of frames")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps (dev pipeline only)")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG guidance scale (dev pipeline only)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=float, default=24.0, help="Frames per second")
    parser.add_argument("--frame-rate", type=float, default=None, help="Alias for --fps")
    parser.add_argument("--output-path", "--output", "-o", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames as images")
    parser.add_argument(
        "--video-encoder",
        type=str,
        choices=["cv2", "ffmpeg"],
        default=os.getenv("LTX_VIDEO_ENCODER", "cv2"),
        help="Video encoder backend for MP4 writing (default: cv2).",
    )
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2", help="Model repository")
    parser.add_argument("--text-encoder-repo", type=str, default=None, help="Text encoder repository")
    parser.add_argument("--checkpoint-path", "--checkpoint", type=str, default=None, help="Path to .safetensors checkpoint (optional)")
    parser.add_argument("--gemma-root", "--text-encoder-path", type=str, default=None, help="Path to Gemma text encoder directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--profile", action="store_true", help="Print a timing breakdown for benchmarking")
    parser.add_argument(
        "--profile-json",
        nargs="?",
        const="auto",
        default=None,
        help="Write a JSON timing breakdown. Optionally provide a path; default writes next to output as *.profile.json.",
    )
    parser.add_argument(
        "--lora",
        action=LoraAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help="LoRA weights to merge (repeatable): --lora path 0.8",
    )
    parser.add_argument(
        "--lora-path",
        action=LoraAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help="Alias for --lora (path and optional strength)",
    )
    parser.add_argument("--enhance-prompt", action="store_true", help="Enhance the prompt using Gemma")
    parser.add_argument(
        "--auto-output-name",
        action="store_true",
        help="Generate a descriptive output filename from the prompt using Gemma",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for prompt enhancement")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for prompt enhancement")
    parser.add_argument(
        "--image",
        "-i",
        action=ImageConditionAction,
        nargs="+",
        metavar="PATH",
        default=[],
        help="Image conditioning. Use: --image path.jpg or --image path.jpg FRAME_IDX STRENGTH (repeatable). FRAME_IDX is a video-frame index and will be mapped to latent frames.",
    )
    parser.add_argument("--condition-image", type=str, default=None, help="Alias for --image (frame 0, strength 1.0)")
    parser.add_argument("--image-strength", type=float, default=1.0, help="Conditioning strength for I2V")
    parser.add_argument("--image-frame-idx", type=int, default=0, help="Frame index to condition for I2V")
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditionAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help="Video conditioning for IC-LoRA: --video-conditioning path.mp4 FRAME_IDX STRENGTH (repeatable). FRAME_IDX is mapped to latent frames.",
    )
    parser.add_argument("--reference-video", type=str, default=None, help="Alias for --video-conditioning (frame 0, strength 1.0)")
    parser.add_argument(
        "--include-reference-in-output",
        action="store_true",
        help="(PyTorch parity) Not implemented in MLX; ignored.",
    )
    parser.add_argument(
        "--distilled-lora",
        action=LoraAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help="LoRA(s) for stage-2 refinement (distilled pipeline only)",
    )
    parser.add_argument(
        "--stage2-model-repo",
        type=str,
        default=None,
        help="(Distilled pipelines) Optional separate model repo/directory for stage-2 refinement.",
    )
    parser.add_argument(
        "--stage2-dev",
        action="store_true",
        help="(Distilled pipelines) Use dev CFG denoiser for stage-2 refinement (requires negative prompt).",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=None,
        help="(Distilled pipelines) Number of denoising steps in stage 1 (1-8).",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=None,
        help="(Distilled pipelines) Number of refinement steps in stage 2 (1-3).",
    )
    parser.add_argument(
        "--sigma-subsample",
        type=str,
        choices=["uniform", "farthest"],
        default=os.getenv("LTX_SIGMA_SUBSAMPLE", "farthest"),
        help="Subsampling method for fixed sigma schedules when reducing steps (default: farthest).",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=["replace", "guide"],
        default="replace",
        help="Image conditioning mode: replace (default) or guide (keyframe-style).",
    )
    parser.add_argument("--tiling", type=str, default="auto",
                        choices=["auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"],
                        help="Tiling mode for VAE decoding")
    parser.add_argument("--stream", action="store_true", help="Stream frames to output as they're decoded")
    parser.add_argument(
        "--audio",
        "-a",
        action="store_true",
        help="Enable synchronized audio generation (default: on; use --skip-audio to disable).",
    )
    parser.add_argument(
        "--audio-mode",
        type=str,
        choices=["auto", "joint", "separate"],
        default=os.getenv("LTX_AUDIO_MODE", "auto"),
        help="Audio strategy when --audio is enabled: auto (default), joint, or separate.",
    )
    parser.add_argument(
        "--audio-model-repo",
        type=str,
        default=os.getenv("LTX_AUDIO_MODEL_REPO") or None,
        help="Optional model repo/directory to use for separate audio transformer weights (dev recommended).",
    )
    parser.add_argument(
        "--audio-steps",
        type=int,
        default=int(os.getenv("LTX_AUDIO_STEPS", "8")),
        help="Number of denoising steps for separate audio generation (1-8).",
    )
    parser.add_argument("--skip-audio", action="store_true", help="Alias for disabling audio generation")
    parser.add_argument("--output-audio", type=str, default=None, help="Output audio path")
    parser.add_argument("--mem-log", action="store_true", help="Log active/cache/peak memory at key stages")
    parser.add_argument("--clear-cache", action="store_true", help="Clear MLX cache after generation")
    parser.add_argument("--cache-limit-gb", type=float, default=None, help="Set MLX cache limit in GB")
    parser.add_argument("--memory-limit-gb", type=float, default=None, help="Set MLX memory limit in GB")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Evaluate latents every N steps (reduces sync overhead; higher uses more memory). Default: auto.",
    )
    # Default audio to ON unless the user opts out. Setting LTX_DEFAULT_AUDIO=0
    # disables this behavior globally, and `--skip-audio` disables per-run.
    env_default_audio = os.getenv("LTX_DEFAULT_AUDIO", "1").lower() in ("1", "true", "yes")
    env_fp32_euler = os.getenv("LTX_FP32_EULER", "").lower() not in ("0", "false", "no")
    env_compile = os.getenv("LTX_COMPILE", "").lower() in ("1", "true", "yes")
    env_compile_shapeless = os.getenv("LTX_COMPILE_SHAPELESS", "").lower() in ("1", "true", "yes")
    env_cfg_batch = os.getenv("LTX_CFG_BATCH", "").lower() in ("1", "true", "yes")
    parser.add_argument(
        "--fp32-euler",
        action=argparse.BooleanOptionalAction,
        default=env_fp32_euler,
        help="Compute Euler update in float32 for numerical stability (slower). Use --no-fp32-euler for speed.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile denoise step for faster repeated execution (auto-enabled for longer runs)",
    )
    parser.add_argument("--compile-shapeless", action="store_true", help="Allow recompile on shape changes")
    parser.add_argument("--no-compile", action="store_true", help="Disable compilation even if env enables it")
    parser.add_argument(
        "--cfg-batch",
        action="store_true",
        help="Batch CFG pos/neg in one forward (auto when cfg_scale > 1; faster, higher memory)",
    )
    parser.add_argument("--no-cfg-batch", action="store_true", help="Disable CFG batching even if env enables it")
    parser.add_argument("--metal-capture", action="store_true", help="Capture Metal GPU trace (.gputrace)")
    parser.add_argument("--metal-capture-path", type=str, default=None, help="Path to .gputrace file")
    parser.add_argument(
        "--metal-capture-phase",
        type=str,
        choices=["denoise", "decode", "all"],
        default="denoise",
        help="Which phase to capture: denoise, decode, or all",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Alias for --steps")
    parser.add_argument("--cfg-guidance-scale", type=float, default=None, help="Alias for --cfg-scale")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Alias for --cfg-scale")
    parser.add_argument("--enable-fp8", action="store_true", help="(PyTorch parity) Not implemented in MLX; ignored.")
    parser.add_argument("--stg-scale", type=float, default=None, help="(PyTorch parity) Not implemented in MLX; ignored.")
    parser.add_argument("--stg-blocks", type=int, nargs="*", default=None, help="(PyTorch parity) Not implemented in MLX; ignored.")
    parser.add_argument("--stg-mode", type=str, choices=["stg_av", "stg_v"], default=None, help="(PyTorch parity) Not implemented in MLX; ignored.")
    args = parser.parse_args()

    if args.profile_json and not args.profile:
        args.profile = True

    # Auto-heuristics for dev-only knobs. Distilled pipelines have fixed schedules,
    # and compile/CFG-batching often add overhead on Apple Metal for short loops.
    is_dev_cli_pipeline = args.pipeline == "dev"
    if args.stage1_steps is None:
        # The distilled schedule contains redundant high-sigma steps; with farthest
        # subsampling, 5 steps is typically very close to 6/8 but faster.
        args.stage1_steps = 5 if args.pipeline == "distilled" else 8
    if args.stage2_steps is None:
        # Default: prefer speed for plain distilled runs; keep historical default
        # (3 steps) for the other two-stage pipelines.
        args.stage2_steps = 1 if args.pipeline == "distilled" else 3
    if args.eval_interval is None:
        env_eval = os.getenv("LTX_EVAL_INTERVAL")
        if env_eval is not None:
            args.eval_interval = int(env_eval)
        else:
            # Fewer evals reduces CPU-GPU sync overhead; distilled runs tolerate a
            # slightly higher interval.
            args.eval_interval = 2 if is_dev_cli_pipeline else 4

    if args.frame_rate is not None:
        args.fps = args.frame_rate
    if args.num_inference_steps is not None:
        args.steps = args.num_inference_steps
    if args.cfg_guidance_scale is not None:
        args.cfg_scale = args.cfg_guidance_scale
    if args.guidance_scale is not None:
        args.cfg_scale = args.guidance_scale
    if args.gemma_root is not None:
        args.text_encoder_repo = args.gemma_root
    if args.eval_interval < 1:
        args.eval_interval = 1
    if args.no_compile:
        args.compile = False
    elif env_compile:
        args.compile = True
    elif not args.compile:
        if is_dev_cli_pipeline and args.steps >= 8:
            args.compile = True
        elif (not is_dev_cli_pipeline) and args.num_frames >= 97 and (args.stage1_steps + args.stage2_steps) >= 5:
            # Distilled pipelines: compile pays off mostly for longer clips and non-trivial step counts.
            args.compile = True
    if not args.compile:
        args.compile_shapeless = False
    elif env_compile_shapeless:
        args.compile_shapeless = True
    if args.no_cfg_batch:
        args.cfg_batch = False
    elif env_cfg_batch:
        args.cfg_batch = True
    elif not args.cfg_batch and is_dev_cli_pipeline and args.cfg_scale > 1.0:
        args.cfg_batch = True

    auto_output_name_model = None
    if args.auto_output_name:
        auto_output_name_model = args.text_encoder_repo

    if args.skip_audio and args.audio:
        console.print("[yellow]⚠️  --skip-audio overrides --audio[/]")
    # If the user configured audio-related knobs, assume they want audio even if
    # they forgot to pass `--audio`. This prevents "missing audio" surprises in
    # UIs that only set `--audio-mode` or `--output-audio`.
    if not args.audio and not args.skip_audio:
        if (
            args.audio_mode != "auto"
            or args.audio_model_repo is not None
            or args.output_audio is not None
            or args.audio_steps != 8
        ):
            args.audio = True
    if env_default_audio and not args.skip_audio:
        args.audio = True
    if args.skip_audio:
        args.audio = False

    if args.enable_fp8:
        console.print("[yellow]⚠️  --enable-fp8 is not supported in MLX (ignored)[/]")
    if args.stg_scale is not None or args.stg_blocks is not None or args.stg_mode is not None:
        console.print("[yellow]⚠️  STG options are not supported in MLX (ignored)[/]")
    if args.include_reference_in_output:
        console.print("[yellow]⚠️  --include-reference-in-output is not supported in MLX (ignored)[/]")

    pipeline = {
        "dev": PipelineType.DEV,
        "distilled": PipelineType.DISTILLED,
        "keyframe": PipelineType.KEYFRAME,
        "ic_lora": PipelineType.IC_LORA,
    }[args.pipeline]

    images = []
    for path, frame_idx, strength in args.image:
        if frame_idx is None:
            frame_idx = args.image_frame_idx
        if strength is None:
            strength = args.image_strength
        images.append((path, frame_idx, strength))
    if args.condition_image:
        images.append((args.condition_image, args.image_frame_idx, args.image_strength))

    video_conditioning = list(args.video_conditioning)
    if args.reference_video:
        video_conditioning.append((args.reference_video, 0, 1.0))

    loras = list(args.lora) + list(args.lora_path)

    generate_video(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        pipeline=pipeline,
        negative_prompt=args.negative_prompt,
        stage2_model_repo=args.stage2_model_repo,
        stage2_dev=args.stage2_dev,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        save_frames=args.save_frames,
        video_encoder=args.video_encoder,
        verbose=args.verbose,
        profile=args.profile,
        profile_json_path=args.profile_json,
        enhance_prompt=args.enhance_prompt,
        enhance_prompt_model=None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=None,
        images=images,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        video_conditionings=video_conditioning,
        distilled_loras=args.distilled_lora,
        conditioning_mode=args.conditioning_mode,
        tiling=args.tiling,
        stream=args.stream,
        audio=args.audio,
        audio_mode=args.audio_mode,
        audio_model_repo=args.audio_model_repo,
        audio_steps=args.audio_steps,
        output_audio_path=args.output_audio,
        mem_log=args.mem_log,
        clear_cache=args.clear_cache,
        cache_limit_gb=args.cache_limit_gb,
        memory_limit_gb=args.memory_limit_gb,
        eval_interval=args.eval_interval,
        compile_step=args.compile,
        compile_shapeless=args.compile_shapeless,
        cfg_batch=args.cfg_batch,
        fp32_euler=args.fp32_euler,
        metal_capture=args.metal_capture,
        metal_capture_path=args.metal_capture_path,
        metal_capture_phase=args.metal_capture_phase,
        loras=loras,
        checkpoint_path=args.checkpoint_path,
        sigma_subsample=args.sigma_subsample,
        auto_output_name=args.auto_output_name,
        output_name_model=auto_output_name_model,
    )


if __name__ == "__main__":
    main()
