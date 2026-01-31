import os
from pathlib import Path

import pytest

from mlx_video.mlx_trainer.trainer import TrainingConfig, MLXTrainer


@pytest.mark.skipif(not os.getenv("LTX_TRAIN_SMOKE"), reason="Set LTX_TRAIN_SMOKE=1 to run heavy trainer test")
def test_trainer_one_step(tmp_path: Path):
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
