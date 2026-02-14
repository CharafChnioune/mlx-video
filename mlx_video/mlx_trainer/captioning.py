from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class CaptionConfig:
    model_id: str = "Salesforce/blip-image-captioning-base"
    max_new_tokens: int = 64
    device: str = "cpu"


@dataclass
class MlxCaptionConfig:
    # Use a smaller MLX VLM by default to keep captioning lightweight.
    model_id: str = "mlx-community/SmolVLM-Instruct-4bit"
    max_tokens: int = 128
    temperature: float = 0.0
    prompt: str = "Describe the image."


class Captioner:
    """Non-MLX captioner placeholder (not supported in MLX-only mode)."""

    def __init__(self, cfg: CaptionConfig | None = None) -> None:
        self.cfg = cfg or CaptionConfig()

    def caption(self, frame: np.ndarray) -> str:  # pragma: no cover
        raise RuntimeError("Captioner is not supported in MLX-only mode; use MlxCaptioner instead.")

    def caption_batch(self, frames: List[np.ndarray]) -> List[str]:  # pragma: no cover
        raise RuntimeError("Captioner is not supported in MLX-only mode; use MlxCaptioner instead.")


class MlxCaptioner:
    def __init__(self, cfg: MlxCaptionConfig | None = None) -> None:
        self.cfg = cfg or MlxCaptionConfig()
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_vlm.chat import load as vlm_load
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mlx_vlm is required for MLX captioning") from exc
        self._model, self._processor = vlm_load(self.cfg.model_id)

    def caption(self, frame: np.ndarray) -> str:
        import mlx.core as mx
        from mlx_vlm.chat import generate_step, get_message_json

        self._load()
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        prompt = self.cfg.prompt
        message = get_message_json(
            self._model.config.model_type,
            prompt,
            role="user",
            skip_image_token=False,
            num_images=1,
        )
        text_prompt = self._processor.apply_chat_template(
            [message], add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[frame],
            padding=True,
            return_tensors="np",
        )
        pixel_values = mx.array(inputs["pixel_values"])
        input_ids = mx.array(inputs["input_ids"])
        mask = mx.array(inputs["attention_mask"])

        detokenizer = self._processor.detokenizer
        detokenizer.reset()

        for token, _ in generate_step(
            input_ids,
            self._model,
            pixel_values,
            mask,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        ):
            if token == self._processor.tokenizer.eos_token_id:
                break
            detokenizer.add_token(token)

        detokenizer.finalize()
        return detokenizer.text.replace("<end_of_utterance>", "").strip()

    def caption_batch(self, frames: List[np.ndarray]) -> List[str]:
        return [self.caption(f) for f in frames]


def get_captioner(backend: str, model_id: str, max_new_tokens: int = 64):
    if backend == "mlx_vlm":
        return MlxCaptioner(MlxCaptionConfig(model_id=model_id, max_tokens=max_new_tokens))
    return Captioner(CaptionConfig(model_id=model_id, max_new_tokens=max_new_tokens))
