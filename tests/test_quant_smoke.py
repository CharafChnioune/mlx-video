import os
import json
from pathlib import Path

import pytest

from mlx_video.convert import convert
from mlx_video.generate import generate_video, PipelineType


@pytest.mark.skipif(not os.getenv("LTX_QUANT_TEST"), reason="Set LTX_QUANT_TEST=1 to run heavy quant test")
def test_quant_conversion_layer_report(tmp_path: Path):
    model_repo = os.getenv("LTX_MODEL_REPO", "Lightricks/LTX-2")
    out_dir = tmp_path / "quant_model_dev"
    convert(
        hf_path=model_repo,
        mlx_path=str(out_dir),
        dtype="bfloat16",
        quantize=True,
        q_bits=8,
        q_group_size=64,
        q_mode="affine",
        quantize_scope="core",
        report_layers=True,
        pipeline="dev",
    )
    assert (out_dir / "quantization.json").exists()
    assert (out_dir / "layer_report.json").exists()
    meta = json.loads((out_dir / "quantization.json").read_text())
    assert meta["bits"] == 8
    assert meta["group_size"] == 64

    out_dev = tmp_path / "quant_dev.mp4"
    generate_video(
        model_repo=str(out_dir),
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
        output_path=str(out_dev),
        verbose=True,
        audio=False,
    )
    assert out_dev.exists()

    out_dir_distilled = tmp_path / "quant_model_distilled"
    convert(
        hf_path=model_repo,
        mlx_path=str(out_dir_distilled),
        dtype="bfloat16",
        quantize=True,
        q_bits=8,
        q_group_size=64,
        q_mode="affine",
        quantize_scope="core",
        report_layers=True,
        pipeline="distilled",
    )
    assert (out_dir_distilled / "quantization.json").exists()
    assert (out_dir_distilled / "layer_report.json").exists()

    out_dist = tmp_path / "quant_distilled.mp4"
    generate_video(
        model_repo=str(out_dir_distilled),
        text_encoder_repo=None,
        prompt="A serene mountain landscape at sunrise, cinematic, soft light.",
        pipeline=PipelineType.DISTILLED,
        height=512,
        width=512,
        num_frames=9,
        num_inference_steps=8,
        cfg_scale=4.0,
        seed=0,
        fps=12,
        output_path=str(out_dist),
        verbose=True,
        audio=False,
    )
    assert out_dist.exists()
