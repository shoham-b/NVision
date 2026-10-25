---
trigger: always_on
---

# Fast, low-token dev loop (SBED / SMC / metrics)

- **Run tests without coverage.** Coverage is opt-in (it made a 3 s test take ~30 s and printed ~20 KB). Plain `pytest` is fast; CI passes `--cov=nvision --cov-report=...` explicitly. For iteration use:
  `uv run pytest tests/<file>.py`
  (add `--cov=nvision --cov-report=term-missing` only when you need coverage)
- **Pick tests by area, do not run the suite.**
  - SBED / focus: `test_sbed_acquire.py`, `test_sbed_thinning.py`, `test_focus_*.py`, `test_refocus.py`, `test_dip_detection.py`
  - SMC / belief: `test_smc_*.py`, `test_unit_cube_belief.py`
  - Metrics: `test_stage_metrics.py`, `test_grid_study_metrics.py`, `test_milestone_metrics_fit_override.py`
- **Never open these** (huge, generated, or irrelevant): `cprofile_output.txt` (~1 MB), `pytest_output.txt`, `notebook.pdf`, `uv.lock`, `artifacts/`, `logs/`, `htmlcov/`, `coverage.xml`, `static/app.js` (~8k lines).
- **Large modules: grep, then read a line range.** Do not read whole files: `spectra/nv_center.py` (2.4k lines), `belief/smc_marginal.py` (1.7k), `cli/run.py` (1.7k), `runner/executor.py` (1.6k), `sim/locs/coarse/generic_sweep_locator.py` (1.4k).
- **Where things live:** SBED `sim/locs/bayesian/sbed_locator.py`; SMC `belief/smc_marginal.py`; focus `belief/focus_window.py`; dips `belief/dip_detection.py`; metrics `metrics/` and `runner/metrics.py`.
- **Verify the pipeline cheaply:** `uv run nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --repeats 1 --runners 1 --dry-run` (never a full `nv run`).
