from __future__ import annotations

import mlx.core as mx

from mlx_video.utils import rms_norm as _mlx_rms_norm


def rms_norm(x: mx.array, weight: mx.array | None = None, eps: float = 1e-6) -> mx.array:
    """MLX RMS normalization.

    If weight is provided, applies it after normalization (scale-only RMSNorm).
    """
    if weight is None:
        return _mlx_rms_norm(x, eps=eps)
    denom = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / denom * weight


def to_denoised(
    sample: mx.array,
    velocity: mx.array,
    sigma: float | mx.array,
    calc_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Convert velocity prediction to denoised sample (x0)."""
    sample_f32 = sample.astype(calc_dtype)
    velocity_f32 = velocity.astype(calc_dtype)
    if isinstance(sigma, (int, float)):
        sigma_f32 = mx.array(sigma, dtype=calc_dtype)
    else:
        sigma_f32 = sigma.astype(calc_dtype)
        while sigma_f32.ndim < velocity_f32.ndim:
            sigma_f32 = mx.expand_dims(sigma_f32, axis=-1)
    result = sample_f32 - sigma_f32 * velocity_f32
    return result.astype(sample.dtype)


def to_velocity(
    sample: mx.array,
    denoised_sample: mx.array,
    sigma: float | mx.array,
    calc_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Convert denoised prediction to velocity (v)."""
    sample_f32 = sample.astype(calc_dtype)
    denoised_f32 = denoised_sample.astype(calc_dtype)
    if isinstance(sigma, (int, float)):
        sigma_f32 = mx.array(sigma, dtype=calc_dtype)
    else:
        sigma_f32 = sigma.astype(calc_dtype)
        while sigma_f32.ndim < denoised_f32.ndim:
            sigma_f32 = mx.expand_dims(sigma_f32, axis=-1)
    result = (sample_f32 - denoised_f32) / sigma_f32
    return result.astype(sample.dtype)
