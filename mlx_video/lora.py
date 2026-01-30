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


def _lora_base_keys(lora_key: str) -> tuple[str, str] | None:
    if not lora_key.endswith(".weight"):
        return None
    if ".lora_A." in lora_key:
        base = lora_key.replace(".lora_A.weight", ".weight")
    elif ".lora_B." in lora_key:
        base = lora_key.replace(".lora_B.weight", ".weight")
    else:
        return None
    # Keep both raw and sanitized forms to match PyTorch and MLX weights
    return base, _sanitize_lora_prefix(base)


def load_lora_state(path: Path) -> Dict[str, mx.array]:
    try:
        return dict(mx.load(str(path)))
    except Exception:
        weights: Dict[str, mx.array] = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))
        return weights


def _iter_lora_pairs(lora_sd: Dict[str, mx.array]) -> Iterable[Tuple[str, str, mx.array, mx.array]]:
    # Yield (base_key_raw, base_key_sanitized, A, B)
    for key in lora_sd.keys():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        key_b = f"{prefix}.lora_B.weight"
        if key_b not in lora_sd:
            continue
        base_keys = _lora_base_keys(key)
        if base_keys is None:
            continue
        base_raw, base_sanitized = base_keys
        yield base_raw, base_sanitized, lora_sd[key], lora_sd[key_b]


def _candidate_weight_keys(base_raw: str, base_sanitized: str) -> Tuple[str, ...]:
    candidates = [base_sanitized, base_raw]
    if base_raw.startswith("diffusion_model."):
        candidates.append(f"model.{base_raw}")
    # Also allow raw weights without "model." prefix (rare)
    if base_sanitized and not base_sanitized.startswith("model."):
        candidates.append(f"diffusion_model.{base_sanitized}")
        candidates.append(f"model.diffusion_model.{base_sanitized}")
    # Preserve order and uniqueness
    seen = set()
    ordered = []
    for key in candidates:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


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
        for base_raw, base_sanitized, A, B in _iter_lora_pairs(lora_sd):
            weight_key = None
            for candidate in _candidate_weight_keys(base_raw, base_sanitized):
                if candidate in updated:
                    weight_key = candidate
                    break
            if weight_key is None:
                skipped += 1
                continue
            w = updated[weight_key]
            # Compute delta = B @ A
            # Cast to float32 for stability, then back to weight dtype
            delta = mx.matmul(B.astype(mx.float32), A.astype(mx.float32)) * spec.strength
            if w.ndim == 4:
                # If convolutional, try to reshape (O, I, H, W) in MLX
                delta = mx.reshape(delta, w.shape)
            delta = delta.astype(w.dtype)
            updated[weight_key] = w + delta
            applied += 1
        if verbose:
            print(f"[LoRA] {spec.path} applied={applied} skipped={skipped}")
        elif applied == 0:
            print(f"[LoRA] Warning: no weights applied for {spec.path}. Check key mapping.")
    return updated


def has_quantized_weights(weights: Dict[str, mx.array]) -> bool:
    return any(k.endswith(".scales") or k.endswith(".biases") for k in weights)
