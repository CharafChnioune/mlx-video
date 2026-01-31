"""Validation sampling helpers (MLX)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from mlx_video.generate import generate_video, PipelineType


@dataclass
class ValidationConfig:
    prompts: list[str]
    model_repo: str = "Lightricks/LTX-2"
    pipeline: str = "dev"
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    width: int = 512
    height: int = 512
    num_frames: int = 33
    steps: int = 20
    cfg_scale: float = 4.0
    fps: float = 24.0
    seed: int = 42
    output_dir: str = "./validation"
    audio: bool = False
    verbose: bool = False


def run_validation(cfg: ValidationConfig) -> list[Path]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    pipeline = PipelineType.DISTILLED if cfg.pipeline == "distilled" else PipelineType.DEV

    for i, prompt in enumerate(cfg.prompts):
        out_path = out_dir / f"sample_{i}.mp4"
        generate_video(
            model_repo=cfg.model_repo,
            text_encoder_repo=None,
            prompt=prompt,
            pipeline=pipeline,
            negative_prompt=cfg.negative_prompt,
            height=cfg.height,
            width=cfg.width,
            num_frames=cfg.num_frames,
            num_inference_steps=cfg.steps,
            cfg_scale=cfg.cfg_scale,
            seed=cfg.seed,
            fps=cfg.fps,
            output_path=str(out_path),
            save_frames=False,
            verbose=cfg.verbose,
            audio=cfg.audio,
        )
        outputs.append(out_path)
    return outputs

