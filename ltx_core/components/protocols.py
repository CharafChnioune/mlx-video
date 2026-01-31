from __future__ import annotations

from typing import Protocol

import mlx.core as mx


class Patchifier(Protocol):
    def patchify(self, latents: mx.array) -> mx.array: ...

    def unpatchify(self, latents: mx.array, output_shape) -> mx.array: ...

    def get_patch_grid_bounds(self, output_shape, device: str | None = None) -> mx.array: ...

    def get_token_count(self, tgt_shape) -> int: ...


class Noiser(Protocol):
    def noise(self, latents: mx.array) -> mx.array: ...


class GuiderProtocol(Protocol):
    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array: ...

    def enabled(self) -> bool: ...


class SchedulerProtocol(Protocol):
    def execute(self, steps: int, **kwargs) -> mx.array: ...


class DiffusionStepProtocol(Protocol):
    def execute(self, sample: mx.array, denoised_sample: mx.array, sigmas: mx.array, step_index: int) -> mx.array: ...
