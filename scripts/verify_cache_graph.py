import logging
import shutil
import traceback
from pathlib import Path

from nvision.cli.runner import _run_combination
from nvision.core.structures import LocatorTask
from nvision.sim import NVCenterSweepLocator
from nvision.sim import cases as sim_cases
from nvision.sim.core import CompositeNoise, CompositeOverFrequencyNoise
from nvision.sim.noises import OverFrequencyGaussianNoise


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

    # Setup test components
    gen_name = "NVCenter-voigt_zeeman"
    # Need to match what _get_generator_category expects (NVCenter)
    # Using a simple generator instance from cases
    gen_obj = next(g[1] for g in sim_cases.generators_basic() if g[0] == gen_name)

    noise_name = "Gauss(0.05)"
    noise_obj = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.05)]))

    strat_name = "NVCenter-Sweep"
    strat_obj = NVCenterSweepLocator(coarse_points=10, refine_points=5)  # Fast settings

    task = LocatorTask(
        task_id="test_task",
        generator_name=gen_name,
        generator=gen_obj,
        noise_name=noise_name,
        noise=noise_obj,
        strategy_name=strat_name,
        strategy=strat_obj,
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
    results1 = _run_combination(task)

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
    results2 = _run_combination(task)

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
