"""Compatibility stub for Gemma tokenizer."""

class LTXVGemmaTokenizer:  # pragma: no cover
    def __init__(self, *_, **__):
        raise NotImplementedError("Tokenizer handled by transformers in mlx_video.models.ltx.text_encoder")
