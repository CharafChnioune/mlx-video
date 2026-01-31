"""MLX keyframe interpolation pipeline wrapper (guide mode)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from mlx_video.generate import PipelineType

from .mlx_bridge import MLXPipelineConfig, run_generate, run_cli


@dataclass
class KeyframeInterpolationPipeline:
    model_repo: str = "Lightricks/LTX-2"
    text_encoder_repo: Optional[str] = None
    height: int = 512
    width: int = 512
    num_frames: int = 33
    steps: int = 8
    cfg_scale: float = 4.0
    seed: int = 42
    fps: float = 24.0
    audio: bool = False
    verbose: bool = False
    stream: bool = False
    tiling: str = "auto"

    def __call__(
        self,
        prompt: str,
        output_path: str = "output.mp4",
        images: Optional[Iterable[tuple[str, int, float]]] = None,
        distilled_lora: Optional[Iterable[tuple[str, float]]] = None,
        negative_prompt: str | None = None,
    ) -> str:
        cfg = MLXPipelineConfig(
            model_repo=self.model_repo,
            text_encoder_repo=self.text_encoder_repo,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            seed=self.seed,
            fps=self.fps,
            audio=self.audio,
            verbose=self.verbose,
            stream=self.stream,
            tiling=self.tiling,
            conditioning_mode="guide",
        )
        return run_generate(
            prompt=prompt,
            pipeline=PipelineType.KEYFRAME,
            cfg=cfg,
            output_path=output_path,
            images=images,
            distilled_loras=distilled_lora,
            negative_prompt=negative_prompt,
        )


def main() -> None:
    run_cli("keyframe")


if __name__ == "__main__":
    main()
