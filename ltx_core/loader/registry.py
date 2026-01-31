"""Minimal registry stubs for MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Registry:
    items: Dict[str, Any]

    def __init__(self):
        self.items = {}

    def register(self, name: str, value: Any):
        self.items[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self.items.get(name, default)


class DummyRegistry(Registry):
    pass


__all__ = ["Registry", "DummyRegistry"]
