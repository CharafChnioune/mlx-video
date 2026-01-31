from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from mlx_video.generate import create_position_grid, create_audio_position_grid
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality
from mlx_video.utils import get_model_path
from .datasets import DummyDataset, PrecomputedDataset, iter_batches, Batch
from .lora import LoRAConfig, inject_lora, freeze_for_lora, export_lora_state, load_lora_state
from .config import load_training_config
from .config_display import print_config
from .hf_hub_utils import push_to_hub
from .progress import TrainingProgress, ProgressStats
from .training_strategies import get_training_strategy
from .timestep_samplers import get_timestep_sampler


@dataclass
class TrainingConfig:
    model_repo: str
    pipeline: str = "dev"  # dev or distilled
    training_mode: str = "full"  # full or lora
    strategy: str = "text_to_video"  # text_to_video or video_to_video
    with_audio: bool = False
    data_root: Optional[str] = None
    data_sources: Optional[dict[str, str]] = None
    batch_size: int = 1
    steps: int = 100
    lr: float = 1e-5
    seed: int = 42
    log_every: int = 1
    output_dir: str = "./checkpoints"
    save_every: int = 100
    checkpoint_keep_last_n: int = -1
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: Optional[Sequence[str]] = None
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer_type: str = "adamw"
    scheduler_type: str = "constant"
    scheduler_params: dict = None
    enable_gradient_checkpointing: bool = False
    first_frame_conditioning_p: float = 0.1
    audio_latents_dir: str = "audio_latents"
    reference_latents_dir: str = "reference_latents"
    timestep_sampling_mode: str = "uniform"
    timestep_sampling_std: float = 1.0
    load_checkpoint: Optional[str] = None
    dummy_width: int = 256
    dummy_height: int = 256
    dummy_num_frames: int = 9
    dummy_prompt_len: int = 256
    debug: bool = False
    validation_prompts: Optional[Sequence[str]] = None
    validation_interval: int = 0
    validation_negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    validation_skip_initial: bool = False
    validation_seed: Optional[int] = None
    validation_width: Optional[int] = None
    validation_height: Optional[int] = None
    validation_num_frames: Optional[int] = None
    validation_steps: Optional[int] = None
    validation_cfg_scale: Optional[float] = None
    validation_fps: Optional[float] = None
    validation_images: Optional[Sequence[str]] = None
    validation_reference_videos: Optional[Sequence[str]] = None
    wandb_enabled: bool = False
    wandb_project: str = "ltx-2-trainer"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[Sequence[str]] = None
    wandb_log_validation: bool = True
    hub_push: bool = False
    hub_model_id: Optional[str] = None
    progress: bool = True
    mixed_precision_mode: str = "bf16"
    load_text_encoder_in_8bit: bool = False
    num_dataloader_workers: int = 0


class MLXTrainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.model_path = self._resolve_model_path(cfg.model_repo)
        self.pipeline = cfg.pipeline
        self.strategy_impl = get_training_strategy(self.cfg)
        if self.strategy_impl.requires_audio and not self.cfg.with_audio:
            if self.cfg.debug:
                print("[trainer] Strategy requires audio; enabling with_audio.")
            self.cfg.with_audio = True
        if self.cfg.debug:
            print_config(self.cfg)
        self._load_dataset()
        self._load_model()
        self._setup_optimizer()
        self._init_wandb()

    def _init_wandb(self) -> None:
        self._wandb = None
        if not self.cfg.wandb_enabled:
            return
        try:
            import wandb  # type: ignore
        except Exception:
            print("[trainer] W&B requested but wandb is not installed.")
            return
        self._wandb = wandb
        self._wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            tags=list(self.cfg.wandb_tags) if self.cfg.wandb_tags else None,
            config={
                "pipeline": self.cfg.pipeline,
                "training_mode": self.cfg.training_mode,
                "strategy": self.cfg.strategy,
                "with_audio": self.cfg.with_audio,
                "steps": self.cfg.steps,
                "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr,
            },
        )

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
        lora_layers = 0
        for _, m in self.model.named_modules():
            if m.__class__.__name__ in {"LoRALinear", "LoRAQuantizedLinear"}:
                lora_layers += 1
        print(f"Linear layers: {linear}, QuantizedLinear layers: {qlinear}")
        print(f"LoRA layers: {lora_layers}")

        trainable_tree = tree_flatten(self.model.trainable_parameters(), destination={})
        total_tree = tree_flatten(self.model.parameters(), destination={})
        trainable = sum(int(np.prod(p.shape)) for p in trainable_tree.values())
        total = sum(int(np.prod(p.shape)) for p in total_tree.values())
        print(f"Trainable params: {trainable:,} / {total:,}")

    def _load_dataset(self) -> None:
        if self.cfg.data_root is None:
            if self.cfg.strategy in {"video_to_video", "ic_lora", "image_to_video"}:
                raise ValueError(
                    f"Strategy '{self.cfg.strategy}' requires --data-root with reference_latents; "
                    "dummy dataset is only supported for text_to_video."
                )
            self.dataset = DummyDataset(
                width=self.cfg.dummy_width,
                height=self.cfg.dummy_height,
                num_frames=self.cfg.dummy_num_frames,
                prompt_sequence_length=self.cfg.dummy_prompt_len,
                with_audio=self.strategy_impl.requires_audio,
            )
        else:
            sources = self.strategy_impl.get_data_sources()
            if self.cfg.data_sources:
                sources = self.cfg.data_sources
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
            lcfg = LoRAConfig(
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.target_modules) if self.cfg.target_modules else None,
            )
            inject_lora(self.model, lcfg)
            freeze_for_lora(self.model)
        else:
            self.model.unfreeze()

        if self.cfg.load_checkpoint and self.cfg.training_mode == "lora":
            ckpt_path = Path(self.cfg.load_checkpoint)
            if ckpt_path.exists():
                try:
                    state = mx.load(str(ckpt_path))
                    load_lora_state(self.model, state)
                    if self.cfg.debug:
                        print(f"[trainer] Loaded LoRA checkpoint: {ckpt_path}")
                except Exception as exc:
                    raise RuntimeError(f"Failed to load LoRA checkpoint: {ckpt_path}") from exc
            else:
                raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_path}")

        self._log_model_stats(weight_files)

    def _setup_optimizer(self) -> None:
        scheduler = self._build_lr_schedule()
        if self.cfg.optimizer_type not in {"adamw", "adamw8bit"}:
            if self.cfg.debug:
                print(f"[trainer] Unsupported optimizer_type={self.cfg.optimizer_type}; using AdamW.")
            opt_type = "adamw"
        else:
            opt_type = self.cfg.optimizer_type
        if opt_type == "adamw8bit":
            if self.cfg.debug:
                print("[trainer] AdamW8bit requested but not available in MLX; using AdamW.")
            opt_type = "adamw"
        if scheduler is None:
            self.optimizer = optim.AdamW(learning_rate=self.cfg.lr)
        else:
            self.optimizer = optim.AdamW(learning_rate=scheduler)

    def _build_lr_schedule(self):
        sched = (self.cfg.scheduler_type or "constant").lower()
        if sched == "constant":
            return None
        total_steps = max(1, int(self.cfg.steps))
        lr0 = float(self.cfg.lr)
        if sched == "linear":
            return optim.linear_schedule(lr0, 0.0, total_steps)
        if sched == "cosine":
            return optim.cosine_decay(lr0, total_steps, end=0.0)
        if self.cfg.debug:
            print(f"[trainer] Unsupported scheduler_type={self.cfg.scheduler_type}; using constant.")
        return None

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

    def _sample_sigmas(self, batch_size: int, seq_len: int) -> mx.array:
        mode = self.cfg.timestep_sampling_mode
        if mode == "shifted_logit_normal":
            min_tokens = 1024
            max_tokens = 4096
            min_shift = 0.95
            max_shift = 2.05
            m = (max_shift - min_shift) / (max_tokens - min_tokens)
            b = min_shift - m * min_tokens
            shift = m * seq_len + b
            normal_samples = mx.random.normal(shape=(batch_size,)) * self.cfg.timestep_sampling_std + shift
            return mx.sigmoid(normal_samples)
        return mx.random.uniform(shape=(batch_size,))

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
        raw_video_embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
        if raw_video_embeds is None:
            raise ValueError("Missing prompt embeddings in conditions (expected video_prompt_embeds or prompt_embeds)")
        video_prompt_embeds = mx.array(raw_video_embeds, dtype=mx.float32)
        audio_prompt_embeds = mx.array(cond.get("audio_prompt_embeds", raw_video_embeds), dtype=mx.float32)

        if video_prompt_embeds.ndim == 2:
            video_prompt_embeds = mx.expand_dims(video_prompt_embeds, axis=0)
        if audio_prompt_embeds.ndim == 2:
            audio_prompt_embeds = mx.expand_dims(audio_prompt_embeds, axis=0)

        prompt_mask_np = cond.get("prompt_attention_mask")
        if prompt_mask_np is None:
            prompt_mask_np = np.ones((video_prompt_embeds.shape[1],), dtype=bool)
        prompt_mask = mx.array(prompt_mask_np)
        if prompt_mask.ndim == 1:
            prompt_mask = mx.expand_dims(prompt_mask, axis=0)

        # noise + sigma
        sigmas = self._sample_sigmas(b, seq_len)
        noise = mx.random.normal(video_latents.shape)
        sigmas_expanded = mx.reshape(sigmas, (b, 1, 1))
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * noise

        conditioning_mask = self._create_first_frame_mask(
            b, num_frames, height, width, p=self.cfg.first_frame_conditioning_p
        )
        noisy_video = mx.where(mx.expand_dims(conditioning_mask, -1), video_latents, noisy_video)

        targets = noise - video_latents
        timesteps = mx.broadcast_to(sigmas.reshape((b, 1)), (b, seq_len))
        timesteps = mx.where(conditioning_mask, mx.zeros_like(timesteps), timesteps)

        positions = create_position_grid(b, num_frames, height, width, fps=fps)

        video_modality = Modality(
            latent=noisy_video,
            timesteps=timesteps.astype(mx.float32),
            positions=positions.astype(mx.float32),
            context=video_prompt_embeds,
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
                context=audio_prompt_embeds,
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
        raw_video_embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
        if raw_video_embeds is None:
            raise ValueError("Missing prompt embeddings in conditions (expected video_prompt_embeds or prompt_embeds)")
        video_prompt_embeds = mx.array(raw_video_embeds, dtype=mx.float32)
        if video_prompt_embeds.ndim == 2:
            video_prompt_embeds = mx.expand_dims(video_prompt_embeds, axis=0)

        prompt_mask_np = cond.get("prompt_attention_mask")
        if prompt_mask_np is None:
            prompt_mask_np = np.ones((video_prompt_embeds.shape[1],), dtype=bool)
        prompt_mask = mx.array(prompt_mask_np)
        if prompt_mask.ndim == 1:
            prompt_mask = mx.expand_dims(prompt_mask, axis=0)

        # conditioning masks
        ref_conditioning = mx.ones((b, ref_seq_len), dtype=mx.bool_)
        target_conditioning = self._create_first_frame_mask(
            b, num_frames, height, width, p=self.cfg.first_frame_conditioning_p
        )
        conditioning_mask = mx.concatenate([ref_conditioning, target_conditioning], axis=1)

        sigmas = self._sample_sigmas(b, ref_seq_len + target_seq_len)
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
            context=video_prompt_embeds,
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
        timestep_sampler = get_timestep_sampler(
            self.cfg.timestep_sampling_mode,
            self.cfg.timestep_sampling_std,
        )
        if self.cfg.debug:
            print(f"[trainer] Timestep sampler: {self.cfg.timestep_sampling_mode} (std={self.cfg.timestep_sampling_std})")
            print(f"[trainer] Strategy: {self.cfg.strategy}")

        def loss_fn(model, batch: Batch):
            inputs = self.strategy_impl.prepare_training_inputs(batch, timestep_sampler)
            v_pred, a_pred = model(inputs.video, inputs.audio)
            return self.strategy_impl.compute_loss(v_pred, a_pred, inputs)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        accum_grads = None
        accum_steps = max(1, int(self.cfg.grad_accum_steps))

        if self.cfg.validation_interval and self.cfg.validation_prompts and not self.cfg.validation_skip_initial:
            self._run_validation(step=0)

        progress = TrainingProgress(self.cfg.steps, enabled=self.cfg.progress and not self.cfg.debug)
        with progress:
            for step, batch in enumerate(iter_batches(self.dataset, self.cfg.batch_size, shuffle=True, seed=self.cfg.seed)):
                if step >= self.cfg.steps:
                    break
                step_start = time.time()
                loss, grads = loss_and_grad(self.model, batch)
                mx.eval(loss, grads)

                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map(lambda a, g: a + g, accum_grads, grads)

                if (step + 1) % accum_steps == 0:
                    scale = 1.0 / accum_steps
                    accum_grads = tree_map(lambda g: g * scale, accum_grads)

                    if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                        flat = tree_flatten(accum_grads, destination={})
                        total = mx.array(0.0)
                        for g in flat.values():
                            total = total + mx.sum(g * g)
                        total_norm = mx.sqrt(total)
                        clip = self.cfg.max_grad_norm / (total_norm + 1e-6)
                        if float(clip) < 1.0:
                            accum_grads = tree_map(lambda g: g * clip, accum_grads)

                    self.optimizer.update(self.model, accum_grads)
                    mx.eval(self.model.parameters())
                    accum_grads = None

                if step % self.cfg.log_every == 0:
                    msg = f"step {step}: loss={float(loss):.6f}"
                    if self.cfg.debug:
                        if hasattr(mx, "get_active_memory"):
                            active = mx.get_active_memory() / (1024 ** 3)
                            peak = mx.get_peak_memory() / (1024 ** 3)
                        elif hasattr(mx, "metal"):
                            active = mx.metal.get_active_memory() / (1024 ** 3)
                            peak = mx.metal.get_peak_memory() / (1024 ** 3)
                        else:
                            active = None
                            peak = None
                        if active is not None and peak is not None:
                            msg += f" | mem_active={active:.2f}GB peak={peak:.2f}GB"
                        msg += f" | step_time={time.time() - step_start:.2f}s"
                    print(msg)
                    if self._wandb is not None:
                        log_data = {"loss": float(loss), "step": step}
                        if self.cfg.debug:
                            log_data["step_time"] = time.time() - step_start
                        self._wandb.log(log_data, step=step)

                if self.cfg.progress and not self.cfg.debug:
                    progress.update(
                        ProgressStats(step=step, total=self.cfg.steps, loss=float(loss), step_time=time.time() - step_start)
                    )

                if self.cfg.save_every and step > 0 and step % self.cfg.save_every == 0:
                    self.save_checkpoint(step)
                    self._prune_checkpoints()

                if (
                    self.cfg.validation_interval
                    and self.cfg.validation_prompts
                    and step > 0
                    and step % self.cfg.validation_interval == 0
                ):
                    self._run_validation(step=step)

        if accum_grads is not None:
            self.optimizer.update(self.model, accum_grads)
            mx.eval(self.model.parameters())
        self.save_checkpoint(self.cfg.steps)
        self._prune_checkpoints()
        if self.cfg.hub_push and self.cfg.hub_model_id:
            try:
                push_to_hub(Path(self.cfg.output_dir), self.cfg.hub_model_id)
            except Exception as exc:
                print(f"[trainer] Hub push failed: {exc}")

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

    def _prune_checkpoints(self) -> None:
        keep_n = self.cfg.checkpoint_keep_last_n
        if keep_n is None or keep_n < 0:
            return
        out_dir = Path(self.cfg.output_dir)
        files = sorted(out_dir.glob("*.safetensors"))
        if len(files) <= keep_n:
            return
        for f in files[:-keep_n]:
            try:
                f.unlink()
            except Exception:
                pass

    def _run_validation(self, step: int) -> None:
        from mlx_video.generate import generate_video, PipelineType

        prompts = list(self.cfg.validation_prompts or [])
        if not prompts:
            return
        images = list(self.cfg.validation_images or [])
        refs = list(self.cfg.validation_reference_videos or [])
        if images and len(images) != len(prompts):
            print("[trainer] Warning: validation images count does not match prompts; missing entries will be ignored.")
        if refs and len(refs) != len(prompts):
            print("[trainer] Warning: validation reference_videos count does not match prompts; missing entries will be ignored.")

        out_dir = Path(self.cfg.output_dir) / "validation" / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_repo = self.cfg.model_repo
        model_path = Path(model_repo)
        if model_path.is_file():
            model_repo = str(model_path.parent)

        width = self.cfg.validation_width or self.cfg.dummy_width
        height = self.cfg.validation_height or self.cfg.dummy_height
        num_frames = self.cfg.validation_num_frames or self.cfg.dummy_num_frames
        steps = self.cfg.validation_steps or 20
        cfg_scale = self.cfg.validation_cfg_scale or 4.0
        fps = self.cfg.validation_fps or 24.0
        seed = self.cfg.validation_seed or self.cfg.seed

        pipe = PipelineType.DISTILLED if self.cfg.pipeline == "distilled" else PipelineType.DEV

        if self.cfg.debug:
            print(f"[trainer] Validation @ step {step}: {len(prompts)} prompt(s)")

        for i, prompt in enumerate(prompts):
            out_path = out_dir / f"sample_{i}.mp4"
            image = images[i] if i < len(images) else None
            ref_video = refs[i] if i < len(refs) else None
            try:
                generate_video(
                    model_repo=model_repo,
                    text_encoder_repo=None,
                    prompt=prompt,
                    pipeline=pipe,
                    negative_prompt=self.cfg.validation_negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    fps=fps,
                    output_path=str(out_path),
                    save_frames=False,
                    image=image,
                    video_conditionings=[(ref_video, 0, 1.0)] if ref_video else None,
                    verbose=self.cfg.debug,
                    audio=self.cfg.with_audio,
                )
                if self._wandb is not None and self.cfg.wandb_log_validation:
                    try:
                        self._wandb.log({f"validation/video_{i}": self._wandb.Video(str(out_path))}, step=step)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[trainer] Validation failed for prompt {i}: {exc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MLX trainer (experimental)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (LTX-2 style)")
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2")
    parser.add_argument("--pipeline", type=str, choices=["dev", "distilled"], default="dev")
    parser.add_argument("--training-mode", type=str, choices=["full", "lora"], default="full")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["text_to_video", "video_to_video", "ic_lora"],
        default="text_to_video",
    )
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--data-sources", type=str, default=None, help="Optional JSON mapping of data sources")
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
    parser.add_argument("--target-modules", type=str, default=None, help="Comma-separated LoRA target module suffixes")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--first-frame-conditioning-p", type=float, default=0.1)
    parser.add_argument("--audio-latents-dir", type=str, default="audio_latents")
    parser.add_argument("--reference-latents-dir", type=str, default="reference_latents")
    parser.add_argument(
        "--timestep-sampling-mode",
        type=str,
        choices=["uniform", "shifted_logit_normal"],
        default="uniform",
    )
    parser.add_argument("--timestep-sampling-std", type=float, default=1.0)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--validation-prompts", type=str, default=None, help="Comma-separated prompts or @file")
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument(
        "--validation-negative-prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
    )
    parser.add_argument("--validation-skip-initial", action="store_true")
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--validation-width", type=int, default=None)
    parser.add_argument("--validation-height", type=int, default=None)
    parser.add_argument("--validation-num-frames", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--validation-cfg-scale", type=float, default=None)
    parser.add_argument("--validation-fps", type=float, default=None)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ltx-2-trainer")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb-log-validation", action="store_true")
    parser.add_argument("--hub-push", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dummy-width", type=int, default=256)
    parser.add_argument("--dummy-height", type=int, default=256)
    parser.add_argument("--dummy-num-frames", type=int, default=9)
    parser.add_argument("--dummy-prompt-len", type=int, default=256)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.config:
        cfg_dict = load_training_config(Path(args.config))
        if args.debug:
            cfg_dict["debug"] = True
        cfg = TrainingConfig(**cfg_dict)
    else:
        target_modules = None
        if args.target_modules:
            target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        validation_prompts = None
        if args.validation_prompts:
            if args.validation_prompts.startswith("@"):
                path = Path(args.validation_prompts[1:])
                validation_prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            else:
                validation_prompts = [p.strip() for p in args.validation_prompts.split(",") if p.strip()]
        wandb_tags = None
        if args.wandb_tags:
            wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        data_sources = None
        if args.data_sources:
            import json

            data_sources = json.loads(args.data_sources)

        cfg = TrainingConfig(
            model_repo=args.model_repo,
            pipeline=args.pipeline,
            training_mode=args.training_mode,
            strategy=args.strategy,
            with_audio=args.with_audio,
            data_root=args.data_root,
            data_sources=data_sources,
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
            target_modules=target_modules,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            first_frame_conditioning_p=args.first_frame_conditioning_p,
            audio_latents_dir=args.audio_latents_dir,
            reference_latents_dir=args.reference_latents_dir,
            timestep_sampling_mode=args.timestep_sampling_mode,
            timestep_sampling_std=args.timestep_sampling_std,
            load_checkpoint=args.load_checkpoint,
            dummy_width=args.dummy_width,
            dummy_height=args.dummy_height,
            dummy_num_frames=args.dummy_num_frames,
            dummy_prompt_len=args.dummy_prompt_len,
            debug=args.debug,
            validation_prompts=validation_prompts,
            validation_interval=args.validation_interval,
            validation_negative_prompt=args.validation_negative_prompt,
            validation_skip_initial=args.validation_skip_initial,
            validation_seed=args.validation_seed,
            validation_width=args.validation_width,
            validation_height=args.validation_height,
            validation_num_frames=args.validation_num_frames,
            validation_steps=args.validation_steps,
            validation_cfg_scale=args.validation_cfg_scale,
            validation_fps=args.validation_fps,
            wandb_enabled=args.wandb_enabled,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
            wandb_log_validation=args.wandb_log_validation,
            hub_push=args.hub_push,
            hub_model_id=args.hub_model_id,
            progress=not args.no_progress,
        )

    trainer = MLXTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
