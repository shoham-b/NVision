from __future__ import annotations

from typing import Any

from rich.progress import ProgressColumn, Task


class DotsColumn(ProgressColumn):
    def render(self, task: Task) -> Any:
        return "." * int(task.completed)
