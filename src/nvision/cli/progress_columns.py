from __future__ import annotations

from rich.progress import ProgressColumn, Task


class DotsColumn(ProgressColumn):
    def render(self, task: Task) -> object:
        return "." * int(task.completed)
