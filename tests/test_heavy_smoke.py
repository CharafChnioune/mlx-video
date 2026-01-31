import os
from pathlib import Path

import pytest

if not os.getenv("LTX_HEAVY"):
    pytest.skip("Set LTX_HEAVY=1 to run heavy integration tests", allow_module_level=True)

from mlx_video.mlx_trainer.trainer import TrainingConfig, MLXTrainer
from mlx_video.generate import generate_video, PipelineType


def test_lora_training_one_step(tmp_path: Path):
    model_repo = os.getenv("LTX_MODEL_REPO", "Lightricks/LTX-2")
    cfg = TrainingConfig(
        model_repo=model_repo,
        pipeline="dev",
        training_mode="lora",
        strategy="text_to_video",
        steps=1,
        batch_size=1,
        output_dir=str(tmp_path),
        debug=True,
    )
    trainer = MLXTrainer(cfg)
    trainer.train()


def test_quant_guardrail(tmp_path: Path):
    model_repo = os.getenv("LTX_QUANT_REPO", "AITRADER/ltx2-dev-8bit-mlx")
    cfg = TrainingConfig(
        model_repo=model_repo,
        pipeline="dev",
        training_mode="full",
        strategy="text_to_video",
        steps=1,
        batch_size=1,
        output_dir=str(tmp_path),
        debug=True,
    )
    with pytest.raises(ValueError):
        MLXTrainer(cfg)


def test_quant_inference_smoke(tmp_path: Path):
    model_repo = os.getenv("LTX_QUANT_REPO", "AITRADER/ltx2-dev-8bit-mlx")
    out = tmp_path / "quant_smoke.mp4"
    generate_video(
        model_repo=model_repo,
        text_encoder_repo=None,
        prompt="A serene mountain landscape at sunrise, cinematic, soft light.",
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
        audio=False,
    )
    assert out.exists()


def test_audio_precompute_smoke(tmp_path: Path):
    sample_dir = os.getenv("LTX_AUDIO_SAMPLE_DIR")
    if not sample_dir:
        pytest.skip("Set LTX_AUDIO_SAMPLE_DIR to a folder with a short video+audio")
    cmd = [
        "python",
        "-m",
        "mlx_video.mlx_trainer.precompute",
        "--input-dir",
        sample_dir,
        "--output-dir",
        str(tmp_path),
        "--with-audio",
        "--debug",
    ]
    import subprocess

    subprocess.run(cmd, check=True)
    assert (Path(tmp_path) / "audio_latents").exists()


def test_decode_latents_audio_smoke(tmp_path: Path):
    latents_dir = os.getenv("LTX_AUDIO_LATENTS_DIR")
    if not latents_dir:
        pytest.skip("Set LTX_AUDIO_LATENTS_DIR to precomputed audio_latents dir")
    cmd = [
        "python",
        "-m",
        "ltx_trainer.scripts.decode_latents",
        latents_dir,
        str(tmp_path),
        "--audio-latents",
        latents_dir,
        "--pipeline",
        "dev",
    ]
    import subprocess

    subprocess.run(cmd, check=True)
