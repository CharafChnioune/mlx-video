"""Transformer model components (MLX wrappers)."""

from mlx_video.models.ltx.transformer import Modality
from mlx_video.models.ltx.ltx import LTXModel, X0Model
from mlx_video.models.ltx.config import LTXModelConfig

# Compatibility constants for legacy loader API
LTXV_MODEL_COMFY_RENAMING_MAP: dict[str, str] = {}
LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP: dict[str, str] = {}
UPCAST_DURING_INFERENCE: set[str] = set()


class LTXModelConfigurator:
    def __init__(self, *_, **__):
        pass

    def build(self) -> LTXModelConfig:
        return LTXModelConfig()


class LTXVideoOnlyModelConfigurator(LTXModelConfigurator):
    pass


class UpcastWithStochasticRounding:
    pass


__all__ = [
    "LTXV_MODEL_COMFY_RENAMING_MAP",
    "LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP",
    "UPCAST_DURING_INFERENCE",
    "LTXModel",
    "LTXModelConfigurator",
    "LTXVideoOnlyModelConfigurator",
    "Modality",
    "UpcastWithStochasticRounding",
    "X0Model",
]
