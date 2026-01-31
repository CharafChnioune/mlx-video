"""Compatibility shim for AV Gemma text encoder."""

from mlx_video.models.ltx.text_encoder import LTX2TextEncoder as AVGemmaTextEncoderModel

AV_GEMMA_TEXT_ENCODER_KEY_OPS = {}

__all__ = ["AVGemmaTextEncoderModel", "AV_GEMMA_TEXT_ENCODER_KEY_OPS"]
