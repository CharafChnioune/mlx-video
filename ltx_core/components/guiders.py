from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ltx_core.components.protocols import GuiderProtocol


def _l2_norm(x: mx.array, axis=None, keepdims: bool = False) -> mx.array:
    return mx.sqrt(mx.sum(x * x, axis=axis, keepdims=keepdims) + 1e-8)


def projection_coef(to_project: mx.array, project_onto: mx.array) -> mx.array:
    batch_size = to_project.shape[0]
    positive_flat = mx.reshape(to_project, (batch_size, -1))
    negative_flat = mx.reshape(project_onto, (batch_size, -1))
    dot_product = mx.sum(positive_flat * negative_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(negative_flat * negative_flat, axis=1, keepdims=True) + 1e-8
    return dot_product / squared_norm


@dataclass(frozen=True)
class CFGGuider(GuiderProtocol):
    scale: float

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        return (self.scale - 1) * (cond - uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class CFGStarRescalingGuider(GuiderProtocol):
    scale: float

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        rescaled_neg = projection_coef(cond, uncond) * uncond
        return (self.scale - 1) * (cond - rescaled_neg)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class STGGuider(GuiderProtocol):
    scale: float

    def delta(self, pos_denoised: mx.array, perturbed_denoised: mx.array) -> mx.array:
        return self.scale * (pos_denoised - perturbed_denoised)

    def enabled(self) -> bool:
        return self.scale != 0.0


@dataclass(frozen=True)
class LtxAPGGuider(GuiderProtocol):
    scale: float
    eta: float = 1.0
    norm_threshold: float = 0.0

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        guidance = cond - uncond
        if self.norm_threshold > 0:
            guidance_norm = _l2_norm(guidance, axis=(-1, -2, -3), keepdims=True)
            scale_factor = mx.minimum(mx.ones_like(guidance), self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor
        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth
        return g_apg * (self.scale - 1)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=False)
class LegacyStatefulAPGGuider(GuiderProtocol):
    scale: float
    eta: float
    norm_threshold: float = 5.0
    momentum: float = 0.0
    running_avg: mx.array | None = None

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        guidance = cond - uncond
        if self.momentum != 0:
            if self.running_avg is None:
                self.running_avg = mx.array(guidance)
            else:
                self.running_avg = self.momentum * self.running_avg + guidance
            guidance = self.running_avg

        if self.norm_threshold > 0:
            guidance_norm = _l2_norm(guidance, axis=(-1, -2, -3), keepdims=True)
            scale_factor = mx.minimum(mx.ones_like(guidance), self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor

        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth
        return g_apg * self.scale

    def enabled(self) -> bool:
        return self.scale != 0.0
