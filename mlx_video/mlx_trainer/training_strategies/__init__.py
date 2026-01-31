from __future__ import annotations

from .base_strategy import TrainingStrategy
from .text_to_video import TextToVideoStrategy
from .video_to_video import VideoToVideoStrategy


def get_training_strategy(cfg) -> TrainingStrategy:
    if cfg.strategy in {"video_to_video", "ic_lora"}:
        return VideoToVideoStrategy(cfg)
    return TextToVideoStrategy(cfg)
