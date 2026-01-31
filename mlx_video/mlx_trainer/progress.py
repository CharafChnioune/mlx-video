from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
except Exception:  # pragma: no cover
    Progress = None
    Console = None


@dataclass
class ProgressStats:
    step: int
    total: int
    loss: float
    step_time: float


class TrainingProgress:
    def __init__(self, total_steps: int, enabled: bool = True) -> None:
        self.total_steps = total_steps
        self.enabled = enabled and Progress is not None
        self._progress: Optional[Progress] = None
        self._task_id: Optional[int] = None
        self._console = Console() if Console is not None else None

    def __enter__(self):
        if not self.enabled:
            return self
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("Training", total=self.total_steps)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._progress is not None:
            self._progress.stop()

    def update(self, stats: ProgressStats) -> None:
        if not self.enabled or self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, completed=stats.step + 1)
        self._progress.refresh()

