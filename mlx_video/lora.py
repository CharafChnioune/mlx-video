from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import mlx.core as mx
from safetensors import safe_open


@dataclass(frozen=True)
class LoraSpec:
    path: Path
    strength: float = 1.0


def _sanitize_lora_prefix(prefix: str) -> str:
    # Remove PyTorch prefixes
    if prefix.startswith("model.diffusion_model."):
        prefix = prefix[len("model.diffusion_model.") :]
    if prefix.startswith("diffusion_model."):
        prefix = prefix[len("diffusion_model.") :]

    # Match MLX sanitized naming (see LTXModel.sanitize)
    prefix = prefix.replace(".to_out.0.", ".to_out.")
    prefix = prefix.replace(".ff.net.0.proj.", ".ff.proj_in.")
    prefix = prefix.replace(".ff.net.2.", ".ff.proj_out.")
    prefix = prefix.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
    prefix = prefix.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
    prefix = prefix.replace(".linear_1.", ".linear1.")
    prefix = prefix.replace(".linear_2.", ".linear2.")
    return prefix


def _lora_base_key(lora_key: str) -> str | None:
    if not lora_key.endswith(".weight"):
        return None
    if ".lora_A." in lora_key:
        base = lora_key.replace(".lora_A.weight", ".weight")
    elif ".lora_B." in lora_key:
        base = lora_key.replace(".lora_B.weight", ".weight")
    else:
        return None
    return _sanitize_lora_prefix(base)


def load_lora_state(path: Path) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key))
    return weights


def _iter_lora_pairs(lora_sd: Dict[str, mx.array]) -> Iterable[Tuple[str, mx.array, mx.array]]:
    # Yield (base_key, A, B)
    for key in lora_sd.keys():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        key_b = f"{prefix}.lora_B.weight"
        if key_b not in lora_sd:
            continue
        base_key = _lora_base_key(key)
        if base_key is None:
            continue
        yield base_key, lora_sd[key], lora_sd[key_b]


def apply_lora_to_weights(
    weights: Dict[str, mx.array],
    lora_specs: Iterable[LoraSpec],
    verbose: bool = False,
) -> Dict[str, mx.array]:
    updated = dict(weights)
    for spec in lora_specs:
        lora_sd = load_lora_state(spec.path)
        applied = 0
        skipped = 0
        for base_key, A, B in _iter_lora_pairs(lora_sd):
            if base_key not in updated:
                skipped += 1
                continue
            w = updated[base_key]
            # Compute delta = B @ A
            # Cast to float32 for stability, then back to weight dtype
            delta = mx.matmul(B.astype(mx.float32), A.astype(mx.float32)) * spec.strength
            if w.ndim == 4:
                # If convolutional, try to reshape (O, I, H, W) in MLX
                delta = mx.reshape(delta, w.shape)
            delta = delta.astype(w.dtype)
            updated[base_key] = w + delta
            applied += 1
        if verbose:
            print(f"[LoRA] {spec.path} applied={applied} skipped={skipped}")
    return updated


def has_quantized_weights(weights: Dict[str, mx.array]) -> bool:
    return any(k.endswith(".scales") or k.endswith(".biases") for k in weights)
