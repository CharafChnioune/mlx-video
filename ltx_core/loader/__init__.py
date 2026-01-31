"""MLX loader shims for compatibility."""

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

LTXV_LORA_COMFY_RENAMING_MAP: dict[str, str] = {}

__all__ = ["SingleGPUModelBuilder", "LTXV_LORA_COMFY_RENAMING_MAP"]
