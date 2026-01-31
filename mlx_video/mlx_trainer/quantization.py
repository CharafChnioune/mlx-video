"""Quantization helpers for MLX trainer.

MLX uses pre-quantized weights. These utilities read quantization metadata and
provide lightweight checks without performing in-place quantization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_quantization_meta(model_path: str | Path) -> Dict[str, Any] | None:
    path = Path(model_path)
    if path.is_file():
        path = path.parent
    meta = path / "quantization.json"
    if not meta.exists():
        return None
    try:
        import json

        return json.loads(meta.read_text())
    except Exception:
        return None


def is_quantized(model_path: str | Path) -> bool:
    return load_quantization_meta(model_path) is not None


def describe_quantization(model_path: str | Path) -> str:
    meta = load_quantization_meta(model_path)
    if not meta:
        return "unquantized"
    bits = meta.get("bits", "?")
    group = meta.get("group_size", "?")
    mode = meta.get("mode", "?")
    scope = meta.get("quantize_scope", meta.get("predicate", "?"))
    return f"{bits}-bit, group={group}, mode={mode}, scope={scope}"

