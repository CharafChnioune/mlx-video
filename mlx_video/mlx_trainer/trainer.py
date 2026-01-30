from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_video.generate import create_position_grid, create_audio_position_grid
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality
from mlx_video.utils import get_model_path
from .datasets import DummyDataset, PrecomputedDataset, iter_batches, Batch
from .lora import LoRAConfig, inject_lora, freeze_for_lora, export_lora_state


@dataclass
class TrainingConfig:
    model_repo: str
    pipeline: str = "dev"  # dev or distilled
    training_mode: str = "full"  # full or lora
    strategy: str = "text_to_video"  # text_to_video or video_to_video
    with_audio: bool = False
    data_root: Optional[str] = None
    batch_size: int = 1
    steps: int = 100
    lr: float = 1e-5
    seed: int = 42
    log_every: int = 1
    output_dir: str = "./checkpoints"
    save_every: int = 100
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    dummy_width: int = 256
    dummy_height: int = 256
    dummy_num_frames: int = 9
    dummy_prompt_len: int = 256
    debug: bool = False


class MLXTrainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.model_path = self._resolve_model_path(cfg.model_repo)
        self.pipeline = cfg.pipeline
        self._load_dataset()
        self._load_model()
        self._setup_optimizer()

    def _resolve_model_path(self, model_repo: str) -> Path:
        path = Path(model_repo)
        if path.exists():
            return path
        return get_model_path(model_repo)

    def _load_weight_files(self) -> Sequence[Path]:
        # Prefer quantized MLX file if present, else fall back to fp16/bf16 file.
        if self.model_path.is_file():
            return [self.model_path]

        pipeline = self.cfg.pipeline
        candidates = [
            self.model_path / f"ltx-2-19b-{pipeline}-mlx.safetensors",
            self.model_path / f"ltx-2-19b-{pipeline}-fp8.safetensors",
            self.model_path / f"ltx-2-19b-{pipeline}-fp4.safetensors",
            self.model_path / f"ltx-2-19b-{pipeline}.safetensors",
        ]
        for c in candidates:
            if c.exists():
                return [c]

        # Sharded weights via index file
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            import json

            with index_path.open("r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            files = sorted({self.model_path / fname for fname in weight_map.values()})
            if files:
                return files

        # Fallback: any safetensors in directory
        files = sorted(self.model_path.glob("*.safetensors"))
        if files:
            return files
        raise FileNotFoundError(f"No safetensors found in {self.model_path}")

    def _log_model_stats(self, weight_files: Sequence[Path]) -> None:
        if not self.cfg.debug:
            return
        print("==> Trainer debug")
        print(f"Model repo/path: {self.model_path}")
        print(f"Weight files: {[str(p) for p in weight_files]}")

        quant_path = self.model_path / "quantization.json"
        if quant_path.exists():
            try:
                import json

                with quant_path.open("r") as f:
                    meta = json.load(f)
                print(f"Quantization meta: {meta}")
            except Exception as exc:
                print(f"Quantization meta read failed: {exc}")

        # Count quantized modules
        qlinear = 0
        linear = 0
        for _, m in self.model.named_modules():
            if isinstance(m, nn.QuantizedLinear):
                qlinear += 1
            elif isinstance(m, nn.Linear):
                linear += 1
        print(f"Linear layers: {linear}, QuantizedLinear layers: {qlinear}")

        from mlx.utils import tree_flatten

        trainable_tree = tree_flatten(self.model.trainable_parameters(), destination={})
        total_tree = tree_flatten(self.model.parameters(), destination={})
        trainable = sum(int(np.prod(p.shape)) for p in trainable_tree.values())
        total = sum(int(np.prod(p.shape)) for p in total_tree.values())
        print(f"Trainable params: {trainable:,} / {total:,}")

    def _load_dataset(self) -> None:
        if self.cfg.data_root is None:
            self.dataset = DummyDataset(
                width=self.cfg.dummy_width,
                height=self.cfg.dummy_height,
                num_frames=self.cfg.dummy_num_frames,
                prompt_sequence_length=self.cfg.dummy_prompt_len,
                with_audio=self.cfg.with_audio,
            )
        else:
            # Expect precomputed dataset format
            if self.cfg.strategy == "video_to_video":
                sources = {"latents": "latents", "conditions": "conditions", "reference_latents": "ref_latents"}
            else:
                sources = {"latents": "latents", "conditions": "conditions"}
                if self.cfg.with_audio:
                    sources["audio_latents"] = "audio_latents"
            self.dataset = PrecomputedDataset(self.cfg.data_root, sources)

    def _load_model(self) -> None:
        model_type = LTXModelType.AudioVideo if self.cfg.with_audio else LTXModelType.VideoOnly
        config_kwargs = dict(
            model_type=model_type,
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            caption_channels=3840,
            rope_type=LTXRopeType.SPLIT,
            double_precision_rope=True,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
        )
        if self.cfg.with_audio:
            config_kwargs.update(
                audio_num_attention_heads=32,
                audio_attention_head_dim=64,
                audio_in_channels=8 * 16,
                audio_out_channels=8 * 16,
                audio_cross_attention_dim=2048,
                audio_positional_embedding_max_pos=[20],
            )
        config = LTXModelConfig(**config_kwargs)
        weight_files = self._load_weight_files()
        self.model = LTXModel.from_pretrained(weight_files, config=config, strict=False)

        # If quantized weights are present, only allow LoRA training
        has_quant_meta = (self.model_path / "quantization.json").exists()
        if has_quant_meta and self.cfg.training_mode != "lora":
            raise ValueError("Quantized weights detected. Use --training-mode lora for quantized training.")

        if self.cfg.training_mode == "lora":
            lcfg = LoRAConfig(rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha, dropout=self.cfg.lora_dropout)
            inject_lora(self.model, lcfg)
            freeze_for_lora(self.model)
        else:
            self.model.unfreeze()

        self._log_model_stats(weight_files)

    def _setup_optimizer(self) -> None:
        self.optimizer = optim.AdamW(learning_rate=self.cfg.lr)

    def _patchify_video(self, latents: mx.array) -> mx.array:
        # latents: [B, C, F, H, W] -> [B, seq, C]
        b, c, f, h, w = latents.shape
        x = mx.transpose(latents, (0, 2, 3, 4, 1))
        return mx.reshape(x, (b, f * h * w, c))

    def _patchify_audio(self, latents: mx.array) -> mx.array:
        # latents: [B, C, T, F] -> [B, T, C*F]
        b, c, t, f = latents.shape
        x = mx.transpose(latents, (0, 2, 1, 3))
        return mx.reshape(x, (b, t, c * f))

    def _create_first_frame_mask(self, b: int, f: int, h: int, w: int, p: float = 0.1) -> mx.array:
        if f <= 0:
            return mx.zeros((b, 0), dtype=mx.bool_)
        first = mx.ones((b, 1, h, w), dtype=mx.bool_)
        if f > 1:
            rest = mx.zeros((b, f - 1, h, w), dtype=mx.bool_)
            mask = mx.concatenate([first, rest], axis=1)
        else:
            mask = first
        mask = mx.reshape(mask, (b, f * h * w))
        if p <= 0:
            return mask * False
        if p >= 1:
            return mask
        keep = mx.random.uniform(shape=(b, 1)) < p
        return mx.where(keep, mask, mx.zeros_like(mask))

    def _prepare_batch_t2v(self, batch: Batch):
        # convert to mx arrays
        lat = batch.latents
        video_latents = mx.array(lat["latents"], dtype=mx.float32)
        num_frames = int(np.array(lat["num_frames"]).reshape(-1)[0])
        height = int(np.array(lat["height"]).reshape(-1)[0])
        width = int(np.array(lat["width"]).reshape(-1)[0])
        fps = float(np.array(lat.get("fps", np.array([24], dtype=np.float32))).reshape(-1)[0])

        video_latents = self._patchify_video(video_latents)
        b, seq_len, _ = video_latents.shape

        cond = batch.conditions
        video_prompt_embeds = mx.array(cond.get("video_prompt_embeds", cond.get("prompt_embeds")), dtype=mx.float32)
        audio_prompt_embeds = mx.array(cond.get("audio_prompt_embeds", video_prompt_embeds), dtype=mx.float32)
        prompt_mask = mx.array(cond.get("prompt_attention_mask", np.ones(video_prompt_embeds.shape[0], dtype=bool)))
        prompt_mask = mx.expand_dims(prompt_mask, axis=0)

        # noise + sigma
        sigmas = mx.random.uniform(shape=(b,))
        noise = mx.random.normal(video_latents.shape)
        sigmas_expanded = mx.reshape(sigmas, (b, 1, 1))
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * noise

        conditioning_mask = self._create_first_frame_mask(b, num_frames, height, width)
        noisy_video = mx.where(mx.expand_dims(conditioning_mask, -1), video_latents, noisy_video)

        targets = noise - video_latents
        timesteps = mx.broadcast_to(sigmas.reshape((b, 1)), (b, seq_len))
        timesteps = mx.where(conditioning_mask, mx.zeros_like(timesteps), timesteps)

        positions = create_position_grid(b, num_frames, height, width, fps=fps)

        video_modality = Modality(
            latent=noisy_video,
            timesteps=timesteps.astype(mx.float32),
            positions=positions.astype(mx.float32),
            context=mx.expand_dims(video_prompt_embeds, 0),
            context_mask=prompt_mask,
            enabled=True,
        )

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None
        if self.cfg.with_audio and batch.audio_latents is not None:
            a_lat = batch.audio_latents
            audio_latents = mx.array(a_lat["latents"], dtype=mx.float32)
            audio_latents = self._patchify_audio(audio_latents)
            ab, at, _ = audio_latents.shape
            audio_noise = mx.random.normal(audio_latents.shape)
            a_sigmas = mx.broadcast_to(sigmas.reshape((b, 1, 1)), audio_latents.shape)
            noisy_audio = (1 - a_sigmas) * audio_latents + a_sigmas * audio_noise
            audio_targets = audio_noise - audio_latents
            audio_timesteps = mx.broadcast_to(sigmas.reshape((b, 1)), (ab, at))
            audio_positions = create_audio_position_grid(ab, at)
            audio_modality = Modality(
                latent=noisy_audio,
                timesteps=audio_timesteps.astype(mx.float32),
                positions=audio_positions.astype(mx.float32),
                context=mx.expand_dims(audio_prompt_embeds, 0),
                context_mask=prompt_mask,
                enabled=True,
            )
            audio_loss_mask = mx.ones((ab, at), dtype=mx.bool_)

        loss_mask = ~conditioning_mask

        return video_modality, audio_modality, targets, audio_targets, loss_mask, audio_loss_mask

    def _prepare_batch_v2v(self, batch: Batch):
        lat = batch.latents
        ref = batch.ref_latents
        if ref is None:
            raise ValueError("video_to_video strategy requires ref_latents")
        target_latents = mx.array(lat["latents"], dtype=mx.float32)
        ref_latents = mx.array(ref["latents"], dtype=mx.float32)

        num_frames = int(np.array(lat["num_frames"]).reshape(-1)[0])
        height = int(np.array(lat["height"]).reshape(-1)[0])
        width = int(np.array(lat["width"]).reshape(-1)[0])
        fps = float(np.array(lat.get("fps", np.array([24], dtype=np.float32))).reshape(-1)[0])

        ref_frames = int(np.array(ref["num_frames"]).reshape(-1)[0])
        ref_height = int(np.array(ref["height"]).reshape(-1)[0])
        ref_width = int(np.array(ref["width"]).reshape(-1)[0])

        target_latents = self._patchify_video(target_latents)
        ref_latents = self._patchify_video(ref_latents)

        b, target_seq_len, _ = target_latents.shape
        ref_seq_len = ref_latents.shape[1]

        cond = batch.conditions
        video_prompt_embeds = mx.array(cond.get("video_prompt_embeds", cond.get("prompt_embeds")), dtype=mx.float32)
        prompt_mask = mx.array(cond.get("prompt_attention_mask", np.ones(video_prompt_embeds.shape[0], dtype=bool)))
        prompt_mask = mx.expand_dims(prompt_mask, axis=0)

        # conditioning masks
        ref_conditioning = mx.ones((b, ref_seq_len), dtype=mx.bool_)
        target_conditioning = self._create_first_frame_mask(b, num_frames, height, width)
        conditioning_mask = mx.concatenate([ref_conditioning, target_conditioning], axis=1)

        sigmas = mx.random.uniform(shape=(b,))
        noise = mx.random.normal(target_latents.shape)
        sigmas_expanded = mx.reshape(sigmas, (b, 1, 1))
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise
        noisy_target = mx.where(mx.expand_dims(target_conditioning, -1), target_latents, noisy_target)
        targets = noise - target_latents

        combined_latents = mx.concatenate([ref_latents, noisy_target], axis=1)

        timesteps = mx.broadcast_to(sigmas.reshape((b, 1)), (b, ref_seq_len + target_seq_len))
        timesteps = mx.where(conditioning_mask, mx.zeros_like(timesteps), timesteps)

        ref_positions = create_position_grid(b, ref_frames, ref_height, ref_width, fps=fps)
        target_positions = create_position_grid(b, num_frames, height, width, fps=fps)
        positions = mx.concatenate([ref_positions, target_positions], axis=2)

        video_modality = Modality(
            latent=combined_latents,
            timesteps=timesteps.astype(mx.float32),
            positions=positions.astype(mx.float32),
            context=mx.expand_dims(video_prompt_embeds, 0),
            context_mask=prompt_mask,
            enabled=True,
        )

        ref_loss_mask = mx.zeros((b, ref_seq_len), dtype=mx.bool_)
        target_loss_mask = ~target_conditioning
        loss_mask = mx.concatenate([ref_loss_mask, target_loss_mask], axis=1)

        return video_modality, None, targets, None, loss_mask, None

    def _loss(self, video_pred, audio_pred, video_targets, audio_targets, video_mask, audio_mask):
        v = (video_pred - video_targets) ** 2
        v = mx.sum(v, axis=-1)
        v = mx.where(video_mask, v, mx.zeros_like(v))
        v_loss = mx.sum(v) / mx.maximum(mx.sum(video_mask), 1)

        a_loss = mx.array(0.0)
        if audio_pred is not None and audio_targets is not None:
            a = (audio_pred - audio_targets) ** 2
            a = mx.sum(a, axis=-1)
            a = mx.where(audio_mask, a, mx.zeros_like(a))
            a_loss = mx.sum(a) / mx.maximum(mx.sum(audio_mask), 1)
        return v_loss + a_loss

    def train(self) -> None:
        mx.random.seed(self.cfg.seed)

        def loss_fn(model, batch: Batch):
            if self.cfg.strategy == "video_to_video":
                video_modality, audio_modality, targets, audio_targets, mask, audio_mask = self._prepare_batch_v2v(batch)
            else:
                video_modality, audio_modality, targets, audio_targets, mask, audio_mask = self._prepare_batch_t2v(batch)
            v_pred, a_pred = model(video_modality, audio_modality)
            return self._loss(v_pred, a_pred, targets, audio_targets, mask, audio_mask)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        for step, batch in enumerate(iter_batches(self.dataset, self.cfg.batch_size, shuffle=True, seed=self.cfg.seed)):
            if step >= self.cfg.steps:
                break
            loss, grads = loss_and_grad(self.model, batch)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters())

            if step % self.cfg.log_every == 0:
                print(f"step {step}: loss={float(loss):.6f}")

            if self.cfg.save_every and step > 0 and step % self.cfg.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.cfg.steps)

    def save_checkpoint(self, step: int) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.training_mode == "lora":
            state = export_lora_state(self.model)
            out_file = out_dir / f"lora_step_{step}.safetensors"
        else:
            from mlx.utils import tree_flatten
            state = tree_flatten(self.model.parameters(), destination={})
            out_file = out_dir / f"transformer_step_{step}.safetensors"
        mx.save_safetensors(str(out_file), state)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MLX trainer (experimental)")
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2")
    parser.add_argument("--pipeline", type=str, choices=["dev", "distilled"], default="dev")
    parser.add_argument("--training-mode", type=str, choices=["full", "lora"], default="full")
    parser.add_argument("--strategy", type=str, choices=["text_to_video", "video_to_video"], default="text_to_video")
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--dummy-width", type=int, default=256)
    parser.add_argument("--dummy-height", type=int, default=256)
    parser.add_argument("--dummy-num-frames", type=int, default=9)
    parser.add_argument("--dummy-prompt-len", type=int, default=256)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    cfg = TrainingConfig(
        model_repo=args.model_repo,
        pipeline=args.pipeline,
        training_mode=args.training_mode,
        strategy=args.strategy,
        with_audio=args.with_audio,
        data_root=args.data_root,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        output_dir=args.output_dir,
        save_every=args.save_every,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dummy_width=args.dummy_width,
        dummy_height=args.dummy_height,
        dummy_num_frames=args.dummy_num_frames,
        dummy_prompt_len=args.dummy_prompt_len,
        debug=args.debug,
    )

    trainer = MLXTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
