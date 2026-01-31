from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import mlx.core as mx


@dataclass
class TimestepSampler:
    """Base class for sampling sigmas/timesteps."""

    def sample_for(self, latents: mx.array, seq_len: int | None = None) -> mx.array:
        raise NotImplementedError


@dataclass
class UniformSampler(TimestepSampler):
    def sample_for(self, latents: mx.array, seq_len: int | None = None) -> mx.array:
        b = latents.shape[0]
        return mx.random.uniform(shape=(b, 1, 1))


@dataclass
class ShiftedLogitNormalSampler(TimestepSampler):
    std: float = 1.0

    def sample_for(self, latents: mx.array, seq_len: int | None = None) -> mx.array:
        b = latents.shape[0]
        if seq_len is None:
            seq_len = latents.shape[1]
        min_tokens = 1024
        max_tokens = 4096
        min_shift = 0.95
        max_shift = 2.05
        m = (max_shift - min_shift) / (max_tokens - min_tokens)
        bias = min_shift - m * min_tokens
        shift = m * seq_len + bias
        normal = mx.random.normal(shape=(b,)) * self.std + shift
        sigmas = mx.sigmoid(normal)
        return mx.reshape(sigmas, (b, 1, 1))


SAMPLERS: Dict[str, TimestepSampler] = {
    "uniform": UniformSampler(),
    "shifted_logit_normal": ShiftedLogitNormalSampler(),
}


def get_timestep_sampler(mode: str, std: float = 1.0) -> TimestepSampler:
    if mode == "shifted_logit_normal":
        return ShiftedLogitNormalSampler(std=std)
    return UniformSampler()
