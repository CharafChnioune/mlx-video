from __future__ import annotations

from dataclasses import asdict


def print_config(cfg) -> None:
    """Pretty-print MLX trainer config."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        print(cfg)
        return

    console = Console()
    table = Table(title="MLX Trainer Config", show_lines=False)
    table.add_column("Key")
    table.add_column("Value")
    for k, v in asdict(cfg).items():
        table.add_row(str(k), str(v))
    console.print(table)
