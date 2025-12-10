import logging
import shutil
from pathlib import Path
from nvision.cli.runner import _run_combination
from nvision.core.types import LocatorTask
from nvision.sim.cases import generators_two, noises_none
from nvision.sim.locs.two_peak.sweep_locator import TwoPeakSweepLocator

logging.basicConfig(level=logging.INFO)


def main():
    out_dir = Path("debug_artifacts")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    gen = generators_two()[0][1]  # TwoPeak-Gaussian
    noise = noises_none()[0][1]  # NoNoise
    strategy = TwoPeakSweepLocator()

    task = LocatorTask(
        generator_name="TwoPeak-Gaussian",
        generator=gen,
        noise_name="NoNoise",
        noise=noise,
        strategy_name="TwoPeak-Sweep",
        strategy=strategy,
        repeats=1,
        seed=123,
        slug="debug-twopeak",
        out_dir=out_dir,
        scans_dir=out_dir / "scans",
        bayes_dir=out_dir / "bayes",
        loc_max_steps=10,
        loc_timeout_s=10,
        use_cache=False,
        cache_dir=out_dir / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        require_cache=False,
        progress_queue=None,
    )

    (task.scans_dir).mkdir(parents=True, exist_ok=True)
    (task.bayes_dir).mkdir(parents=True, exist_ok=True)

    print("Running combination...")
    results = _run_combination(task)
    print("Results:", results)


if __name__ == "__main__":
    main()
