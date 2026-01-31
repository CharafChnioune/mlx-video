import gc
import logging
from dataclasses import replace

import mlx.core as mx
from tqdm import tqdm

from ltx_core.components.noisers import Noiser
from ltx_core.components.protocols import DiffusionStepProtocol, GuiderProtocol
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.model.video_vae import VideoEncoder
from ltx_core.text_encoders.gemma import GemmaTextEncoderModelBase
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.media_io import decode_image, load_image_conditioning, resize_aspect_ratio_preserving
from ltx_pipelines.utils.types import (
    DenoisingFunc,
    DenoisingLoopFunc,
    PipelineComponents,
)


def get_device() -> str:
    return "cpu"


def cleanup_memory() -> None:
    gc.collect()
    mx.clear_cache()


def image_conditionings_by_replacing_latent(
    images: list[tuple[str, int, float]],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: mx.Dtype,
    device: str,
) -> list[ConditioningItem]:
    conditionings = []
    for image_path, frame_idx, strength in images:
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=strength,
                latent_idx=frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[tuple[str, int, float]],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: mx.Dtype,
    device: str,
) -> list[ConditioningItem]:
    conditionings = []
    for image_path, frame_idx, strength in images:
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=frame_idx, strength=strength)
        )
    return conditionings


def euler_denoising_loop(
    sigmas: mx.array,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
) -> tuple[LatentState, LatentState]:
    for step_idx, _ in enumerate(tqdm(list(range(int(sigmas.shape[0] - 1))))):
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        video_state = replace(video_state, latent=stepper.execute(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.execute(audio_state.latent, denoised_audio, sigmas, step_idx))

    return (video_state, audio_state)


def gradient_estimating_euler_denoising_loop(
    sigmas: mx.array,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    ge_gamma: float = 2.0,
) -> tuple[LatentState, LatentState]:
    previous_audio_velocity = None
    previous_video_velocity = None

    def update_velocity_and_sample(
        noisy_sample: mx.array, denoised_sample: mx.array, sigma: float, previous_velocity: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    for step_idx, _ in enumerate(tqdm(list(range(int(sigmas.shape[0] - 1))))):
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if float(sigmas[step_idx + 1]) == 0:
            return replace(video_state, latent=denoised_video), replace(audio_state, latent=denoised_audio)

        previous_video_velocity, denoised_video = update_velocity_and_sample(
            video_state.latent, denoised_video, float(sigmas[step_idx]), previous_video_velocity
        )
        previous_audio_velocity, denoised_audio = update_velocity_and_sample(
            audio_state.latent, denoised_audio, float(sigmas[step_idx]), previous_audio_velocity
        )

        video_state = replace(video_state, latent=stepper.execute(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.execute(audio_state.latent, denoised_audio, sigmas, step_idx))

    return (video_state, audio_state)


def noise_video_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: mx.Dtype,
    device: str,
    noise_scale: float = 1.0,
    initial_latent: mx.array | None = None,
) -> tuple[LatentState, VideoLatentTools]:
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)
    video_state = create_noised_state(
        tools=video_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return video_state, video_tools


def noise_audio_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: mx.Dtype,
    device: str,
    noise_scale: float = 1.0,
    initial_latent: mx.array | None = None,
) -> tuple[LatentState, AudioLatentTools]:
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
    audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return audio_state, audio_tools


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: mx.Dtype,
    device: str,
    noise_scale: float = 1.0,
    initial_latent: mx.array | None = None,
) -> LatentState:
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)

    noise = noiser.noise(state.latent) * noise_scale
    state = replace(state, latent=state.latent + noise)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: mx.array, denoise_mask: mx.array, clean: mx.array) -> mx.array:
    return (denoised * denoise_mask + clean.astype(mx.float32) * (1 - denoise_mask)).astype(denoised.dtype)


def modality_from_latent_state(
    state: LatentState, context: mx.array, sigma: float | mx.array, enabled: bool = True
) -> Modality:
    return Modality(
        enabled=enabled,
        latent=state.latent,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
    )


def timesteps_from_mask(denoise_mask: mx.array, sigma: float | mx.array) -> mx.array:
    return denoise_mask * sigma


def simple_denoising_func(
    video_context: mx.array, audio_context: mx.array, transformer: X0Model
) -> DenoisingFunc:
    def simple_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: mx.array, step_index: int
    ) -> tuple[mx.array, mx.array]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, video_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, audio_context, sigma)

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return denoised_video, denoised_audio

    return simple_denoising_step


def guider_denoising_func(
    guider: GuiderProtocol,
    v_context_p: mx.array,
    v_context_n: mx.array,
    a_context_p: mx.array,
    a_context_n: mx.array,
    transformer: X0Model,
) -> DenoisingFunc:
    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: mx.array, step_index: int
    ) -> tuple[mx.array, mx.array]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
        pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        if guider.enabled():
            neg_video = modality_from_latent_state(video_state, v_context_n, sigma)
            neg_audio = modality_from_latent_state(audio_state, a_context_n, sigma)

            neg_denoised_video, neg_denoised_audio = transformer(video=neg_video, audio=neg_audio, perturbations=None)

            denoised_video = denoised_video + guider.delta(denoised_video, neg_denoised_video)
            denoised_audio = denoised_audio + guider.delta(denoised_audio, neg_denoised_audio)

        return denoised_video, denoised_audio

    return guider_denoising_step


def denoise_audio_video(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: mx.array,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: mx.Dtype,
    device: str,
    noise_scale: float = 1.0,
    initial_video_latent: mx.array | None = None,
    initial_audio_latent: mx.array | None = None,
) -> tuple[LatentState, LatentState]:
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )
    audio_state, audio_tools = noise_audio_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_audio_latent,
    )

    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
    )

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    text = text.translate(_UNICODE_REPLACEMENTS)

    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoderModelBase,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = resize_aspect_ratio_preserving(image, image_long_side)
        image = mx.array(image)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
