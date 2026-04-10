import logging
import shutil
import traceback
from pathlib import Path

from nvision import CombinationGrid, LocatorTask, run_task


def main():
    logging.basicConfig(level=logging.INFO)

    out_dir = Path("artifacts_verify_cache")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cache_dir = out_dir / "cache"
    cache_dir.mkdir()

    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir()
    (graphs_dir / "scans").mkdir()
    (graphs_dir / "bayes").mkdir()

    # Pick one deterministic combination from the grid.
    grid = CombinationGrid()
    combo = next(grid.iter(filter_category="NVCenter", filter_strategy="SimpleSweep"))

    task = LocatorTask(
        task_id="test_task",
        combination=combo,
        repeats=1,
        seed=123,
        slug="test_slug",
        out_dir=out_dir,
        scans_dir=graphs_dir / "scans",
        bayes_dir=graphs_dir / "bayes",
        loc_max_steps=20,
        loc_timeout_s=60,
        use_cache=True,
        cache_dir=cache_dir,
        log_queue=None,
        log_level=logging.DEBUG,
        ignore_cache_strategy=None,
        require_cache=False,
        progress_queue=None,
    )

    print("\n--- RUN 1: Populating Cache ---")
    results1 = run_task(task)

    # Verify we got results
    if not results1:
        print("Run 1 produced no results!")
        return

    # Verify graph file exists
    expected_graph = graphs_dir / "scans" / "test_slug_r1.html"
    if not expected_graph.exists():
        print(f"FAILED: Graph file {expected_graph} not created in Run 1.")
        return
    print(f"Run 1 success. Graph created at {expected_graph}")

    # Clear graphs
    print("\n--- Clearing Graphs ---")
    shutil.rmtree(graphs_dir)
    graphs_dir.mkdir()
    (graphs_dir / "scans").mkdir()
    (graphs_dir / "bayes").mkdir()

    if expected_graph.exists():
        print(f"FAILED: Graph file {expected_graph} still exists after clear.")
        return

    print("\n--- RUN 2: Restoring from Cache ---")
    # Run again with same task (use_cache=True)
    results2 = run_task(task)

    if not results2:
        print("Run 2 produced no results (cache miss or empty cache?)")
        return

    if not expected_graph.exists():
        print(f"FAILED: Graph file {expected_graph} was NOT restored from cache.")
        return

    # Check content roughly
    content = expected_graph.read_text("utf-8")
    if len(content) < 100:
        print("FAILED: Restored graph content seems too short/empty.")
        return

    print(f"SUCCESS: Graph file restored from cache! Size: {len(content)} bytes.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
