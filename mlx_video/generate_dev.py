"""Dev-pipeline helper exports.

Historically, some tests and downstream code imported scheduler/utility
functions from `mlx_video.generate_dev`. The implementation was consolidated
into `mlx_video.generate`, but we keep this module as a thin re-export layer
for backwards compatibility.
"""

from __future__ import annotations

from .generate import (  # noqa: F401
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_SAMPLE_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    PipelineType,
    cfg_delta,
    compute_audio_frames,
    create_audio_position_grid,
    create_position_grid,
    generate_video,
    ltx2_scheduler,
)

def generate_video_dev(
    model_repo: str,
    text_encoder_repo: str,
    prompt: str,
    *,
    tiling: str = "none",
    **kwargs,
):
    """Back-compat wrapper for the dev pipeline.

    The unified implementation lives in `mlx_video.generate.generate_video`.
    We keep this thin wrapper so older imports/tests can rely on a stable API.

    Note: This wrapper defaults `tiling="none"` for parity with older behavior.
    """

    return generate_video(
        model_repo=model_repo,
        text_encoder_repo=text_encoder_repo,
        prompt=prompt,
        pipeline=PipelineType.DEV,
        tiling=tiling,
        **kwargs,
    )

__all__ = [
    "AUDIO_LATENTS_PER_SECOND",
    "AUDIO_SAMPLE_RATE",
    "DEFAULT_NEGATIVE_PROMPT",
    "cfg_delta",
    "compute_audio_frames",
    "create_audio_position_grid",
    "create_position_grid",
    "generate_video_dev",
    "ltx2_scheduler",
]
