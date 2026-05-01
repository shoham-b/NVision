1. **Fix `n_pairs` division by zero**:
   - In `nvision/viz/bayesian.py`, when calculating the grid for Fisher Information pairs (`n_cols`, `n_rows`), handle the case where `n_pairs == 0` by returning early.
2. **Revert incorrect Ruff fixes**:
   - In `nvision/sim/locs/coarse/sweep_locator.py` and `nvision/sim/locs/refocus/window.py`, ensure variables `best_point_norm` and `observed_span` are properly kept if they were removed by an unsafe ruff fix. Wait, I'll just check if they are still there and if I need to fix them manually.
