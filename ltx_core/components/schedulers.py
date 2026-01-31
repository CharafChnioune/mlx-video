from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import scipy.stats
import mlx.core as mx

from ltx_core.components.protocols import SchedulerProtocol

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler(SchedulerProtocol):
    """Default scheduler for LTX-2 diffusion sampling (MLX)."""

    def execute(
        self,
        steps: int,
        latent: mx.array | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> mx.array:
        tokens = int(np.prod(latent.shape[2:])) if latent is not None else MAX_SHIFT_ANCHOR
        sigmas = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)

        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        power = 1
        if np.any(sigmas != 0):
            exp_shift = math.exp(sigma_shift)
            sigmas = np.where(
                sigmas != 0,
                exp_shift / (exp_shift + (1 / sigmas - 1) ** power),
                sigmas,
            )

        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return mx.array(sigmas, dtype=mx.float32)


class LinearQuadraticScheduler(SchedulerProtocol):
    """Linear then quadratic schedule."""

    def execute(self, steps: int, threshold_noise: float = 0.025, linear_steps: int | None = None, **_kwargs) -> mx.array:
        if steps == 1:
            return mx.array([1.0, 0.0], dtype=mx.float32)

        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_sigma_schedule = []
        if quadratic_steps > 0:
            quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
            linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
            const = quadratic_coef * (linear_steps**2)
            quadratic_sigma_schedule = [
                quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, steps)
            ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
        return mx.array(sigma_schedule, dtype=mx.float32)


class BetaScheduler(SchedulerProtocol):
    """Beta distribution schedule."""

    shift = 2.37
    timesteps_length = 10000

    def execute(self, steps: int, alpha: float = 0.6, beta: float = 0.6) -> mx.array:
        model_sampling_sigmas = _precalculate_model_sampling_sigmas(self.shift, self.timesteps_length)
        total_timesteps = len(model_sampling_sigmas) - 1
        ts = 1 - np.linspace(0, 1, steps, endpoint=False)
        ts = np.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps).tolist()
        ts = list(dict.fromkeys(ts))

        sigmas = [float(model_sampling_sigmas[int(t)]) for t in ts] + [0.0]
        return mx.array(sigmas, dtype=mx.float32)


@lru_cache(maxsize=5)
def _precalculate_model_sampling_sigmas(shift: float, timesteps_length: int) -> np.ndarray:
    timesteps = np.arange(1, timesteps_length + 1, 1) / timesteps_length
    return np.array([flux_time_shift(shift, 1.0, t) for t in timesteps])


def flux_time_shift(mu: float, sigma: float, t: float) -> float:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
