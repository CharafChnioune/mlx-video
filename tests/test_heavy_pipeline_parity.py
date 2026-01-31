import os
from pathlib import Path

import pytest

if not os.getenv("LTX_PIPELINE_SMOKE"):
    pytest.skip("Set LTX_PIPELINE_SMOKE=1 to run heavy pipeline tests", allow_module_level=True)

from mlx_video.generate import generate_video, PipelineType


def _model_repo():
    return os.getenv("LTX_MODEL_REPO", "Lightricks/LTX-2")


def test_dev_pipeline(tmp_path: Path):
    out = tmp_path / "dev.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A cinematic mountain landscape at sunrise.",
        pipeline=PipelineType.DEV,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=10,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        verbose=True,
    )
    assert out.exists()


def test_distilled_pipeline(tmp_path: Path):
    out = tmp_path / "distilled.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A cinematic mountain landscape at sunrise.",
        pipeline=PipelineType.DISTILLED,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=8,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        verbose=True,
    )
    assert out.exists()


def test_keyframe_pipeline(tmp_path: Path):
    sample_image = os.getenv("LTX_SAMPLE_IMAGE")
    if not sample_image:
        pytest.skip("Set LTX_SAMPLE_IMAGE to run keyframe test")
    out = tmp_path / "keyframe.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A serene landscape.",
        pipeline=PipelineType.KEYFRAME,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=10,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        images=[(sample_image, 0, 1.0)],
        conditioning_mode="guide",
        verbose=True,
    )
    assert out.exists()


def test_multi_image_conditioning(tmp_path: Path):
    img1 = os.getenv("LTX_SAMPLE_IMAGE")
    img2 = os.getenv("LTX_SAMPLE_IMAGE2")
    if not img1 or not img2:
        pytest.skip("Set LTX_SAMPLE_IMAGE and LTX_SAMPLE_IMAGE2 to run multi-image test")
    out = tmp_path / "multi_image.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A serene landscape.",
        pipeline=PipelineType.KEYFRAME,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=10,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        images=[(img1, 0, 0.8), (img2, 8, 0.8)],
        conditioning_mode="guide",
        verbose=True,
    )
    assert out.exists()


def test_ic_lora_pipeline(tmp_path: Path):
    sample_video = os.getenv("LTX_SAMPLE_VIDEO")
    if not sample_video:
        pytest.skip("Set LTX_SAMPLE_VIDEO to run IC-LoRA test")
    out = tmp_path / "ic_lora.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A cinematic landscape.",
        pipeline=PipelineType.IC_LORA,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=10,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        video_conditionings=[(sample_video, 0, 1.0)],
        verbose=True,
    )
    assert out.exists()


def test_stage2_distilled_lora(tmp_path: Path):
    lora_path = os.getenv("LTX_DISTILLED_LORA")
    if not lora_path:
        pytest.skip("Set LTX_DISTILLED_LORA to run distilled stage-2 LoRA test")
    out = tmp_path / "distilled_lora.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A cinematic landscape.",
        pipeline=PipelineType.DISTILLED,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=8,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        distilled_loras=[(lora_path, 1.0)],
        verbose=True,
    )
    assert out.exists()


def test_stream_audio_mux(tmp_path: Path):
    out = tmp_path / "stream_audio.mp4"
    generate_video(
        model_repo=_model_repo(),
        text_encoder_repo=None,
        prompt="A cinematic landscape.",
        pipeline=PipelineType.DEV,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=10,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out),
        stream=True,
        audio=True,
        verbose=True,
    )
    assert out.exists()
