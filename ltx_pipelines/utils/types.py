from typing import Protocol

import mlx.core as mx

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.types import LatentState
from ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS


class PipelineComponents:
    """
    Container class for pipeline components used throughout the LTX pipelines.

    Attributes:
        dtype: Default MLX dtype for tensors in the pipeline.
        device: Target device identifier (kept for API compatibility).
        video_scale_factors: Scale factors (T, H, W) for VAE latent space.
        video_latent_channels: Number of channels in the video latent representation.
        video_patchifier: Patchifier instance for video latents.
        audio_patchifier: Patchifier instance for audio latents.
    """

    def __init__(self, dtype: mx.Dtype, device: str):
        self.dtype = dtype
        self.device = device

        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.video_latent_channels = VIDEO_LATENT_CHANNELS

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)


class DenoisingFunc(Protocol):
    """Protocol for a denoising function used in the LTX pipeline."""

    def __call__(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: mx.array,
        step_index: int,
    ) -> tuple[mx.array, mx.array]: ...


class DenoisingLoopFunc(Protocol):
    """Protocol for a denoising loop function used in the LTX pipeline."""

    def __call__(
        self,
        sigmas: mx.array,
        video_state: LatentState,
        audio_state: LatentState,
        stepper: DiffusionStepProtocol,
    ) -> tuple[mx.array, mx.array]: ...
