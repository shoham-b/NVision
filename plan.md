1.  **Refactor `Viz.plot_all_metrics` in `nvision/viz/__init__.py`**
    *   (Already completed via `run_in_bash_session` running `patch_viz.py`)
2.  **Verify codebase with formatting, linting and testing**
    *   (Already completed via `run_in_bash_session` executing `uv run ruff format`, `uv run ruff check`, and the full test suite `uv run pytest`)
3.  **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
4.  **Submit the change**
    *   Use the `submit` tool to create the pull request.
    *   Title: "⚡ Bolt: [Optimize Polars dataframe filtering with partition_by]"
    *   Description with:
        * 💡 What: Refactored `Viz.plot_all_metrics` to use `df_loc.partition_by` instead of looping and `.filter()`.
        * 🎯 Why: Replaced O(M*N) complexity loop with O(N) partitioning grouping logic for large dataframe performance boosts.
        * 📊 Impact: Substantial speedup when generating visualizations for large metric datasets containing many combinations.
        * 🔬 Measurement: Observe reduction in runtime when generating multiple experiment metrics visualizations.
