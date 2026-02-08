from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, Any

import mlx.core as mx
import mlx.nn as nn
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


def _strip_known_prefixes(path: str) -> str:
    if path.startswith("model.diffusion_model."):
        return path[len("model.diffusion_model.") :]
    if path.startswith("diffusion_model."):
        return path[len("diffusion_model.") :]
    if path.startswith("model."):
        return path[len("model.") :]
    return path


def _to_module_path(weight_key: str) -> str:
    """Convert a weight key like `...to_q.weight` into a module path `...to_q`."""
    key = _strip_known_prefixes(weight_key)
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    return key


def _get_by_path(root: Any, path: str) -> Any:
    cur = root
    if not path:
        return cur
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _set_by_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts:
        raise ValueError("Empty module path")
    parent_path = ".".join(parts[:-1])
    name = parts[-1]
    parent = _get_by_path(root, parent_path) if parent_path else root
    if name.isdigit():
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def _looks_like_linear(m: Any) -> bool:
    # LoRA adapters are defined for linear projections (2D weight matrices).
    w = getattr(m, "weight", None)
    if w is not None and getattr(w, "ndim", 0) == 2:
        return True
    scales = getattr(m, "scales", None)
    if scales is not None and getattr(scales, "ndim", 0) == 2:
        return True
    return False


class LoRAAdapter(nn.Module):
    """Runtime LoRA wrapper for MLX modules.

    This avoids in-place merges that break quantized checkpoints (4/8-bit) and can
    introduce severe artifacts ("snow") if the quantization layout is not identical.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._pairs: list[tuple[mx.array, mx.array, float]] = []

    def add(self, A: mx.array, B: mx.array, strength: float) -> None:
        # Store raw tensors; compute in float32 at call time for stability.
        self._pairs.append((A, B, float(strength)))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if not self._pairs:
            return y

        x_f32 = x.astype(mx.float32)
        delta = None
        for A, B, s in self._pairs:
            # A: (r, in), B: (out, r)
            a_f32 = A.astype(mx.float32)
            b_f32 = B.astype(mx.float32)
            # (..., in) @ (in, r) -> (..., r) -> (..., out)
            d = mx.matmul(mx.matmul(x_f32, mx.transpose(a_f32)), mx.transpose(b_f32)) * s
            delta = d if delta is None else (delta + d)

        return y + delta.astype(y.dtype)


def apply_lora_to_model(
    model: nn.Module,
    lora_specs: Iterable[LoraSpec],
    verbose: bool = False,
) -> nn.Module:
    """Attach LoRA adapters to a loaded model in-place."""
    available = {p for p, _ in model.named_modules()}

    total_applied = 0
    for spec in lora_specs:
        lora_sd = load_lora_state(spec.path)
        applied = 0
        skipped = 0

        for base_raw, base_sanitized, A, B in _iter_lora_pairs(lora_sd):
            target_path = None
            for candidate in _candidate_weight_keys(base_raw, base_sanitized):
                mp = _to_module_path(candidate)
                if mp in available:
                    target_path = mp
                    break
            if target_path is None:
                skipped += 1
                continue

            try:
                mod = _get_by_path(model, target_path)
            except Exception:
                skipped += 1
                continue

            if not _looks_like_linear(mod):
                skipped += 1
                continue

            if isinstance(mod, LoRAAdapter):
                adapter = mod
            else:
                adapter = LoRAAdapter(mod)
                _set_by_path(model, target_path, adapter)
                available.add(target_path)

            adapter.add(A, B, spec.strength)
            applied += 1

        total_applied += applied
        if verbose:
            print(f"[LoRA] runtime attach {spec.path} applied={applied} skipped={skipped}")
        elif applied == 0:
            print(f"[LoRA] Warning: no runtime adapters attached for {spec.path}. Check key mapping.")

    if verbose:
        print(f"[LoRA] runtime total applied pairs={total_applied}")
    return model
