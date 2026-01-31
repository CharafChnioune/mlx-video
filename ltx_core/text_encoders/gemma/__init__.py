"""MLX Gemma text encoder aliases."""

from mlx_video.models.ltx.text_encoder import LTX2TextEncoder as AVGemmaTextEncoderModel


class GemmaTextEncoderModelBase:  # compatibility stub
    pass


__all__ = ["AVGemmaTextEncoderModel", "GemmaTextEncoderModelBase"]
