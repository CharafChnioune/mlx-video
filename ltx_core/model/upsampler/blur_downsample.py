"""MLX blur downsample stub."""

import mlx.core as mx


class BlurDownsample:  # pragma: no cover - compatibility shim
    def __init__(self, *_, **__):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        return x
