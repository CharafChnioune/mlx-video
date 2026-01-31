import numpy as np

from mlx_video.mlx_trainer import precompute


def test_select_bucket_picks_nearest():
    frames = np.zeros((73, 480, 832, 3), dtype=np.uint8)
    buckets = [
        (73, 480, 832),
        (65, 512, 512),
    ]
    assert precompute._select_bucket(frames, buckets) == (73, 480, 832)


def test_match_frame_count_trim_and_pad():
    frames = np.zeros((81, 10, 10, 3), dtype=np.uint8)
    trimmed = precompute._match_frame_count(frames, 65)
    assert trimmed.shape[0] == 65

    frames = np.zeros((49, 10, 10, 3), dtype=np.uint8)
    padded = precompute._match_frame_count(frames, 65)
    assert padded.shape[0] == 65
    assert np.all(padded[-1] == frames[-1])
