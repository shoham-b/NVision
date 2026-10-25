Always use `uv` when invoking pip and when running python.
Never run python directly (e.g., `python script.py`); always use the `uv run nv <command>` interface.
Do NOT run tests or lint checks on files that were not modified by the task requested. Only target modified files/tests.
All temporary scratch files, check scripts, or refactoring scripts MUST be placed inside the `scratch/` directory. All debug or reproduction scripts MUST be placed inside the `debug/` directory. Do not write scratch or debug files in the repository root.
Focus on **Lorentzian signals with Gaussian noise** for `sbed_locator.py` and `smc_marginal.py`, which keeps particles on the unit cube and rescales public summaries (`focus_window.py` holds the locator-owned drive-frequency focus policy). Distinguish clearly between unit-normalized (`[0, 1]`) and physically scaled parameters.
When adding or modifying an algorithmic constant for fine-tuning, ask the user if they want to put it in `.env`. If so, expose it in `nvision/cli/defaults.py` and add it to both `.env` and `.env.example`.
Before doing anything complex, read `AGENTS.md` in the repository root for rules regarding UI, SMC, SBED, CLI usage, and server configuration.
If you change core math/logic, explicitly tell the user whether to clear cache/re-run or just `uv run nv render` (via `nvision/cli/render.py`).
Always document expected array shapes in docstrings for multi-dimensional arrays.
Always use Plotly for graphs, not Matplotlib.
Coverage is opt-in: iterate with `uv run pytest tests/<file>.py` (CI passes --cov explicitly). Run only the tests for the area you touched (SBED/focus: test_sbed_*, test_focus_*, test_refocus, test_dip_detection; SMC: test_smc_*, test_unit_cube_belief; metrics: test_stage_metrics, test_grid_study_metrics, test_milestone_metrics_fit_override).
Never open cprofile_output.txt, pytest_output.txt, notebook.pdf, uv.lock, artifacts/, logs/, htmlcov/, coverage.xml, or static/app.js. For large modules (spectra/nv_center.py, belief/smc_marginal.py, cli/run.py, runner/executor.py, generic_sweep_locator.py) grep first, then read a line range.
See AGENTS.md section 3b for the full fast dev loop.

Guardrails: stay in scope (no drive-by refactors or config/CI/dependency changes); treat artifacts/, data/, logs/ and the shared cache as read-only (no nv cache clean, no full nv run; use --dry-run); git workflow is restricted to creating/working on dedicated task branches, committing, pushing to that branch, and opening PRs (never touch main directly; no reset --hard, force-push, rebase, or --no-verify); confirm before destructive or remote-facing commands; say whether results are stale and whether the physics fingerprint needs a bump after algorithm changes; fail fast with no silent fallbacks; ask when ambiguous. See AGENTS.md section 3c.
