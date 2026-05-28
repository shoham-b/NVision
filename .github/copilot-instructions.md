Always use `uv` when invoking pip and when running python.
Never run python directly (e.g., `python script.py`); always use the `uv run nv <command>` interface.
Do NOT run tests or lint checks on files that were not modified by the task requested. Only target modified files/tests.
All temporary scratch files, check scripts, or refactoring scripts MUST be placed inside the `scratch/` directory. All debug or reproduction scripts MUST be placed inside the `debug/` directory. Do not write scratch or debug files in the repository root.
Focus on **Lorentzian signals with Gaussian noise** for `sbed_locator.py` and `smc_marginal.py`, wrapped by `unit_cube_smc_marginal.py` for re-scaling. Distinguish clearly between unit-normalized (`[0, 1]`) and physically scaled parameters.
When adding or modifying an algorithmic constant for fine-tuning, ask the user if they want to put it in `.env`. If so, expose it in `nvision/cli/defaults.py` and add it to both `.env` and `.env.example`.
Before doing anything complex, read `AGENTS.md` in the repository root for rules regarding UI, SMC, SBED, CLI usage, and server configuration.
If you change core math/logic, explicitly tell the user whether to clear cache/re-run or just `uv run nv render` (via `nvision/cli/render.py`).
Always document expected array shapes in docstrings for multi-dimensional arrays.
Always use Plotly for graphs, not Matplotlib.