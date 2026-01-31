"""MLX audio VAE wrappers."""

from mlx_video.models.ltx.audio_vae import AudioDecoder, Vocoder

# Minimal compat constants/aliases
AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER: dict[str, str] = {}
AUDIO_VAE_DECODER_COMFY_KEYS_FILTER: dict[str, str] = {}
VOCODER_COMFY_KEYS_FILTER: dict[str, str] = {}


class AudioEncoderConfigurator:
    def __init__(self, *_, **__):
        pass


class AudioDecoderConfigurator:
    def __init__(self, *_, **__):
        pass


class VocoderConfigurator:
    def __init__(self, *_, **__):
        pass


AudioEncoder = AudioDecoder

__all__ = [
    "AudioDecoder",
    "AudioEncoder",
    "Vocoder",
    "AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER",
    "AUDIO_VAE_DECODER_COMFY_KEYS_FILTER",
    "VOCODER_COMFY_KEYS_FILTER",
    "AudioEncoderConfigurator",
    "AudioDecoderConfigurator",
    "VocoderConfigurator",
]
