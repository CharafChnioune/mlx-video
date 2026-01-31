"""MLX bridge utilities for LTX-2 pipeline wrappers.

These wrappers provide a PyTorch-like API but execute MLX inference via
`mlx_video.generate.generate_video`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from mlx_video.generate import generate_video, PipelineType


@dataclass
class MLXPipelineConfig:
    model_repo: str = "Lightricks/LTX-2"
    text_encoder_repo: Optional[str] = None
    height: int = 512
    width: int = 512
    num_frames: int = 33
    steps: int = 40
    cfg_scale: float = 4.0
    seed: int = 42
    fps: float = 24.0
    audio: bool = False
    verbose: bool = False
    stream: bool = False
    tiling: str = "auto"
    conditioning_mode: str = "replace"


def _ensure_list(value):
    if value is None:
        return []
    return list(value)


def _normalize_loras(loras: Optional[Iterable[tuple[str, float]]]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for item in _ensure_list(loras):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), float(item[1])))
        elif isinstance(item, (list, tuple)) and len(item) == 1:
            out.append((str(item[0]), 1.0))
        else:
            out.append((str(item), 1.0))
    return out


def _normalize_images(images: Optional[Iterable[tuple[str, int, float]]]) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    for item in _ensure_list(images):
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((str(item[0]), int(item[1]), float(item[2])))
        elif isinstance(item, (list, tuple)) and len(item) == 1:
            out.append((str(item[0]), 0, 1.0))
        else:
            out.append((str(item), 0, 1.0))
    return out


def _normalize_video_conditions(video_conditionings: Optional[Iterable[tuple[str, int, float]]]) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    for item in _ensure_list(video_conditionings):
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((str(item[0]), int(item[1]), float(item[2])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), 0, float(item[1])))
        else:
            out.append((str(item), 0, 1.0))
    return out


def run_generate(
    prompt: str,
    pipeline: PipelineType,
    cfg: MLXPipelineConfig,
    output_path: str,
    images: Optional[Iterable[tuple[str, int, float]]] = None,
    video_conditionings: Optional[Iterable[tuple[str, int, float]]] = None,
    loras: Optional[Iterable[tuple[str, float]]] = None,
    distilled_loras: Optional[Iterable[tuple[str, float]]] = None,
    negative_prompt: str | None = None,
) -> str:
    generate_video(
        model_repo=cfg.model_repo,
        text_encoder_repo=cfg.text_encoder_repo,
        prompt=prompt,
        pipeline=pipeline,
        negative_prompt=negative_prompt or "",
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.num_frames,
        num_inference_steps=cfg.steps,
        cfg_scale=cfg.cfg_scale,
        seed=cfg.seed,
        fps=cfg.fps,
        output_path=output_path,
        save_frames=False,
        verbose=cfg.verbose,
        image=None,
        images=_normalize_images(images),
        video_conditionings=_normalize_video_conditions(video_conditionings),
        conditioning_mode=cfg.conditioning_mode,
        tiling=cfg.tiling,
        stream=cfg.stream,
        audio=cfg.audio,
        loras=_normalize_loras(loras),
        distilled_loras=_normalize_loras(distilled_loras),
    )
    return output_path


def run_cli(default_pipeline: str) -> None:
    """Dispatch to mlx_video.generate CLI with a default pipeline if not provided."""
    import sys
    if "--pipeline" not in sys.argv:
        sys.argv.extend(["--pipeline", default_pipeline])
    from mlx_video.generate import main as _main
    _main()

