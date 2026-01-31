from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class GaussianNoiser:
    """Gaussian noiser for MLX."""

    seed: Optional[int] = None

    def noise(self, latents: mx.array) -> mx.array:
        if self.seed is not None:
            mx.random.seed(self.seed)
        return mx.random.normal(latents.shape).astype(latents.dtype)
