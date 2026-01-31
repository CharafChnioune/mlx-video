import numpy as np

from mlx_video.mlx_trainer.datasets import PrecomputedDataset


def test_normalize_patchified_latents():
    data = {
        "latents": np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2 * 3 * 4, 5),
        "num_frames": np.array([2], dtype=np.int32),
        "height": np.array([3], dtype=np.int32),
        "width": np.array([4], dtype=np.int32),
    }
    normalized = PrecomputedDataset._normalize_video_latents(data)
    latents = normalized["latents"]
    assert latents.shape == (5, 2, 3, 4)
