from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _get_nested(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _normalize_target_modules(targets: list[str] | None) -> list[str] | None:
    if not targets:
        return targets
    normalized: list[str] = []
    for t in targets:
        t = t.replace("to_out.0", "to_out")
        t = t.replace("ff.net.0.proj", "ff.proj_in")
        t = t.replace("ff.net.2", "ff.proj_out")
        t = t.replace("audio_ff.net.0.proj", "audio_ff.proj_in")
        t = t.replace("audio_ff.net.2", "audio_ff.proj_out")
        normalized.append(t)
    return normalized


def load_training_config(path: Path):
    """Load an LTX-2 style YAML config and map to MLX TrainingConfig.

    Unsupported fields are ignored with a short warning to keep parity flexible.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load training configs.") from exc

    raw = yaml.safe_load(path.read_text()) or {}
    model_cfg = raw.get("model", {})
    lora_cfg = raw.get("lora", {})
    training_cfg = raw.get("training_strategy", {})
    optim_cfg = raw.get("optimization", {})
    data_cfg = raw.get("data", {})
    checkpoint_cfg = raw.get("checkpoints", {})
    flow_cfg = raw.get("flow_matching", {})
    validation_cfg = raw.get("validation", {})
    hub_cfg = raw.get("hub", {})
    wandb_cfg = raw.get("wandb", {})
    accel_cfg = raw.get("acceleration", {})

    model_path = model_cfg.get("model_path", "Lightricks/LTX-2")
    training_mode = model_cfg.get("training_mode", "lora")
    load_checkpoint = model_cfg.get("load_checkpoint")

    strategy = training_cfg.get("name", "text_to_video")
    first_frame_p = training_cfg.get("first_frame_conditioning_p", 0.1)
    with_audio = training_cfg.get("with_audio", False)
    audio_latents_dir = training_cfg.get("audio_latents_dir", "audio_latents")
    reference_latents_dir = training_cfg.get("reference_latents_dir", "reference_latents")

    lr = optim_cfg.get("learning_rate", 1e-5)
    steps = optim_cfg.get("steps", 100)
    batch_size = optim_cfg.get("batch_size", 1)
    grad_accum = optim_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = optim_cfg.get("max_grad_norm", 1.0)
    optimizer_type = optim_cfg.get("optimizer_type", "adamw")
    scheduler_type = optim_cfg.get("scheduler_type", "constant")
    scheduler_params = optim_cfg.get("scheduler_params", {}) or {}
    enable_gradient_checkpointing = optim_cfg.get("enable_gradient_checkpointing", False)

    data_root = data_cfg.get("preprocessed_data_root")
    data_sources = data_cfg.get("data_sources")
    num_dataloader_workers = data_cfg.get("num_dataloader_workers", 0)

    save_every = checkpoint_cfg.get("interval") or 0
    keep_last_n = checkpoint_cfg.get("keep_last_n", -1)
    output_dir = raw.get("output_dir", "./checkpoints")
    seed = raw.get("seed", 42)

    timestep_sampling_mode = flow_cfg.get("timestep_sampling_mode", "uniform")
    timestep_params = flow_cfg.get("timestep_sampling_params", {}) or {}
    timestep_sampling_std = timestep_params.get("std", 1.0)

    lora_rank = lora_cfg.get("rank", 8)
    lora_alpha = lora_cfg.get("alpha", 16.0)
    lora_dropout = lora_cfg.get("dropout", 0.0)
    target_modules = _normalize_target_modules(lora_cfg.get("target_modules"))

    cfg = dict(
        model_repo=model_path,
        pipeline=raw.get("pipeline", "dev"),
        training_mode=training_mode,
        strategy=strategy,
        with_audio=with_audio,
        data_root=data_root,
        data_sources=data_sources,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        seed=seed,
        log_every=raw.get("log_every", 1),
        output_dir=output_dir,
        save_every=save_every,
        checkpoint_keep_last_n=keep_last_n,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        grad_accum_steps=grad_accum,
        max_grad_norm=max_grad_norm,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        first_frame_conditioning_p=first_frame_p,
        audio_latents_dir=audio_latents_dir,
        reference_latents_dir=reference_latents_dir,
        timestep_sampling_mode=timestep_sampling_mode,
        timestep_sampling_std=timestep_sampling_std,
        load_checkpoint=load_checkpoint,
        mixed_precision_mode=accel_cfg.get("mixed_precision_mode", "bf16"),
        load_text_encoder_in_8bit=accel_cfg.get("load_text_encoder_in_8bit", False),
        num_dataloader_workers=num_dataloader_workers,
        validation_prompts=validation_cfg.get("prompts") or None,
        validation_interval=validation_cfg.get("interval") or 0,
        validation_negative_prompt=validation_cfg.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
        validation_skip_initial=validation_cfg.get("skip_initial_validation", False),
        validation_seed=validation_cfg.get("seed"),
        validation_width=validation_cfg.get("width"),
        validation_height=validation_cfg.get("height"),
        validation_num_frames=validation_cfg.get("num_frames"),
        validation_steps=validation_cfg.get("steps"),
        validation_cfg_scale=validation_cfg.get("cfg_scale"),
        validation_fps=validation_cfg.get("fps"),
        validation_images=validation_cfg.get("images"),
        validation_reference_videos=validation_cfg.get("reference_videos"),
        wandb_enabled=wandb_cfg.get("enabled", False),
        wandb_project=wandb_cfg.get("project", "ltx-2-trainer"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_tags=wandb_cfg.get("tags"),
        wandb_log_validation=wandb_cfg.get("log_validation_videos", True),
        hub_push=hub_cfg.get("push_to_hub", False),
        hub_model_id=hub_cfg.get("hub_model_id"),
    )

    if raw.get("model", {}).get("text_encoder_path"):
        print("[trainer] Note: text_encoder_path is ignored in MLX trainer (expects precomputed embeddings).")
    if raw.get("validation"):
        print("[trainer] Note: validation prompts/interval/seed/size are supported; other keys are ignored.")
    if accel_cfg.get("quantization"):
        print("[trainer] Note: acceleration.quantization is ignored; use quantized model weights instead.")
    if enable_gradient_checkpointing:
        print("[trainer] Note: gradient checkpointing is not implemented in MLX trainer (ignored).")
    if optimizer_type not in {"adamw", "adamw8bit"}:
        print(f"[trainer] Warning: optimizer_type '{optimizer_type}' not supported; using adamw.")
    if scheduler_type not in {"constant", "linear", "cosine"}:
        print(f"[trainer] Warning: scheduler_type '{scheduler_type}' not supported; using constant.")
    if num_dataloader_workers:
        print("[trainer] Note: num_dataloader_workers is ignored in MLX trainer (synchronous loading).")

    supported_strategies = {"text_to_video", "video_to_video", "ic_lora"}
    if cfg["strategy"] not in supported_strategies:
        print(f"[trainer] Warning: unsupported strategy '{cfg['strategy']}', defaulting to text_to_video.")
        cfg["strategy"] = "text_to_video"

    if cfg["training_mode"] not in {"full", "lora"}:
        print(f"[trainer] Warning: unsupported training_mode '{cfg['training_mode']}', defaulting to lora.")
        cfg["training_mode"] = "lora"

    return cfg
