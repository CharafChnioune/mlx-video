"""Compatibility types for loader APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class LoraPathStrengthAndSDOps:
    path: Path
    strength: float
    sd_ops: Mapping[str, str] | None = None


__all__ = ["LoraPathStrengthAndSDOps"]
