import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

from mlx_video.lora import LoraSpec, apply_lora_to_weights


def test_apply_lora_simple(tmp_path):
    # base weight
    base = mx.array(np.eye(4, dtype=np.float32))
    weights = {"linear.weight": base}

    # lora A (r=2, in=4), B (out=4, r=2)
    A = np.ones((2, 4), dtype=np.float32)
    B = np.ones((4, 2), dtype=np.float32)

    lora_path = tmp_path / "lora.safetensors"
    save_file(
        {
            "diffusion_model.linear.lora_A.weight": A,
            "diffusion_model.linear.lora_B.weight": B,
        },
        str(lora_path),
    )

    out = apply_lora_to_weights(weights, [LoraSpec(lora_path, 0.5)])
    # delta = B @ A = shape (4,4) all 2s => scaled by 0.5 => all 1s
    expected = np.eye(4, dtype=np.float32) + np.ones((4, 4), dtype=np.float32)
    np.testing.assert_allclose(out["linear.weight"].astype(mx.float32), expected, rtol=1e-5, atol=1e-5)
