"""Misc MLX trainer utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import mlx.core as mx


def save_checkpoint(params: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), params)


def load_checkpoint(path: str | Path) -> dict:
    return dict(mx.load(str(Path(path))))


def parse_prompts(value: Optional[str]) -> list[str]:
    if not value:
        return []
    if value.startswith("@"):
        return [line.strip() for line in Path(value[1:]).read_text().splitlines() if line.strip()]
    return [p.strip() for p in value.split(",") if p.strip()]


def flatten_prompts(prompts: Iterable[str]) -> list[str]:
    return [p.strip() for p in prompts if p.strip()]

