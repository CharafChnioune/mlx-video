from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] | None = None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.weight = base.weight
        if hasattr(base, "bias") and base.bias is not None:
            self.bias = base.bias
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else None
        # LoRA matrices
        in_dim = base.weight.shape[1]
        out_dim = base.weight.shape[0]
        # A: (r, in), B: (out, r)
        self.lora_A = mx.random.normal((rank, in_dim)) * 0.01
        self.lora_B = mx.zeros((out_dim, rank))

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        if self.rank > 0:
            x_in = self.dropout(x) if self.dropout is not None else x
            # (B, ..., in) @ (r, in)^T -> (B, ..., r)
            # then @ (out, r)^T -> (B, ..., out)
            delta = (x_in @ self.lora_A.T) @ self.lora_B.T
            y = y + delta * self.scaling
        return y


class LoRAQuantizedLinear(nn.Module):
    def __init__(self, base: nn.QuantizedLinear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else None

        # Infer dims from quantized metadata
        out_dim = int(base.scales.shape[0])
        group_size = int(getattr(base, "group_size", 64))
        in_dim = int(base.scales.shape[1] * group_size)

        self.lora_A = mx.random.normal((rank, in_dim)) * 0.01
        self.lora_B = mx.zeros((out_dim, rank))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if self.rank > 0:
            x_in = self.dropout(x) if self.dropout is not None else x
            delta = (x_in @ self.lora_A.T) @ self.lora_B.T
            y = y + delta * self.scaling
        return y


def _should_apply(name: str, target_modules: Iterable[str] | None) -> bool:
    if not target_modules:
        return True
    return any(name.endswith(t) or f".{t}" in name for t in target_modules)


def inject_lora(model: nn.Module, config: LoRAConfig) -> None:
    if config.target_modules is None:
        config.target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "ff.proj_in",
            "ff.proj_out",
            "audio_ff.proj_in",
            "audio_ff.proj_out",
            "audio_attn1",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ]

    def _set_module(root: nn.Module, path: str, new_module: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for part in parts[:-1]:
            if isinstance(parent, dict):
                key = part
                if key not in parent and key.isdigit():
                    key = int(key)
                parent = parent[key]
            else:
                parent = getattr(parent, part)
        last = parts[-1]
        if isinstance(parent, dict):
            key = last
            if key not in parent and key.isdigit():
                key = int(key)
            parent[key] = new_module
        else:
            setattr(parent, last, new_module)

    for path, module in model.named_modules():
        if path == "":
            continue
        if isinstance(module, nn.Linear) and _should_apply(path, config.target_modules):
            _set_module(model, path, LoRALinear(module, config.rank, config.alpha, config.dropout))
        elif isinstance(module, nn.QuantizedLinear) and _should_apply(path, config.target_modules):
            _set_module(model, path, LoRAQuantizedLinear(module, config.rank, config.alpha, config.dropout))


def freeze_for_lora(model: nn.Module) -> None:
    model.freeze()
    # Unfreeze LoRA params only
    def _unfreeze(_, m):
        if isinstance(m, (LoRALinear, LoRAQuantizedLinear)):
            m.unfreeze(keys=["lora_A", "lora_B"])
    model.apply_to_modules(_unfreeze)


def export_lora_state(model: nn.Module) -> dict:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAQuantizedLinear)):
            key_prefix = f"diffusion_model.{name}"
            state[f"{key_prefix}.lora_A.weight"] = module.lora_A
            state[f"{key_prefix}.lora_B.weight"] = module.lora_B
    return state
