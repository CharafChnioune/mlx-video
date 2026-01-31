"""MLX PixelShuffle wrapper."""

import mlx.core as mx


class PixelShuffleND:  # pragma: no cover - compatibility shim
    def __init__(self, upscale_factor: int = 2):
        self.upscale_factor = upscale_factor

    def __call__(self, x: mx.array) -> mx.array:
        # Only supports 4D BCHW input
        if x.ndim != 4:
            return x
        b, c, h, w = x.shape
        r = self.upscale_factor
        if c % (r * r) != 0:
            return x
        out_c = c // (r * r)
        x = mx.reshape(x, (b, out_c, r, r, h, w))
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        return mx.reshape(x, (b, out_c, h * r, w * r))
