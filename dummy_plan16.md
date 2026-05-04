Wait, my current `HEAD` is `cc58e7a`!
I reset my branch to before my `feat: add recalculate command and functionality` commit because I made a mistake merging the previous branch.
I need to rewrite the code for `recalculate` command!

The CI failure I got the third time was actually the CI running on the SECOND push I made when I tried to push `numpy`. The GitHub actions ran for `b3af95e` or `62f1e3e`.
But the current state of my local workspace is completely reset to the BEGINNING of the task.

I need to quickly:
1. Re-run `modify_metrics.py`
2. Re-run `modify_render.py`
3. Add `numpy>=2.0.0` to `pyproject.toml` dependencies
4. `uv sync`
5. `git add .`
6. `git commit -m "feat: add recalculate command"`
7. `submit`
