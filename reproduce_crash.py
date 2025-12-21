import logging
import sys
from pathlib import Path
from nvision.cli.runner import _run_combination
from nvision.core.types import LocatorTask
from nvision.sim import cases as sim_cases
from nvision.sim.noises.over_frequency import OverFrequencyGaussianNoise

# Setup basic logging to stdout
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


# Mock objects
class MockQueue:
    def put(self, item):
        pass


def reproduction():
    out_dir = Path("reproduce_artifacts")
    out_dir.mkdir(exist_ok=True)

    # Pick a combination that likely failed: NVCenter-ProjectBayesian with some noise.
    # From log: NVCenter-voigt_zeeman/Heavy/NVCenter-ProjectBayesian

    gen_map = dict(sim_cases.generators_basic())
    gen_name = "NVCenter-voigt_zeeman"
    gen_obj = gen_map[gen_name]

    noise_obj = OverFrequencyGaussianNoise(sigma=0.05)  # Simple noise first
    noise_name = "Gauss(0.05)"

    # Strategy
    # We need to find the strategy object.
    # nvision.sim.cases.locators defines them.
    from nvision.sim.locs.nv_center.project_bayesian_locator import NVCenterProjectBayesianLocator

    strat_obj = NVCenterProjectBayesianLocator()
    strat_name = "NVCenter-ProjectBayesian"

    task = LocatorTask(
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
        scans_dir=out_dir / "scans",
        bayes_dir=out_dir / "bayes",
        loc_max_steps=10,
        loc_timeout_s=60,
        use_cache=False,
        cache_dir=out_dir / "cache",
        log_queue=None,
        log_level=logging.DEBUG,
        ignore_cache_strategy=None,
        require_cache=False,
        progress_queue=MockQueue(),
        task_id=0,
    )

    print("Running combination...")
    _run_combination(task)
    print("Done.")


if __name__ == "__main__":
    reproduction()
