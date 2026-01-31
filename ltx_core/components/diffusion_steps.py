from __future__ import annotations

import mlx.core as mx

from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.utils import to_velocity


class EulerDiffusionStep(DiffusionStepProtocol):
    def execute(self, sample: mx.array, denoised_sample: mx.array, sigmas: mx.array, step_index: int) -> mx.array:
        velocity = to_velocity(sample, denoised_sample, sigmas[step_index])
        dt = sigmas[step_index + 1] - sigmas[step_index]
        return (sample.astype(mx.float32) + velocity.astype(mx.float32) * dt).astype(sample.dtype)
