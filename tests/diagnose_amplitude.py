"""Check amplitude-linewidth degeneracy in the posterior for single Lorentzian."""

import random

from nvision import (
    CoreExperiment,
    MultiPeakCoreGenerator,
    SequentialBayesianExperimentDesignLocator,
    one_peak_lorentzian_belief,
)
from nvision.sim.gen.peak_spec import LORENTZIAN


def main() -> None:
    # Use single Lorentzian peak
    gen = MultiPeakCoreGenerator(count=1, peak_configs=[LORENTZIAN])
    rng = random.Random(42)
    true_signal = gen.generate(rng)
    tp = true_signal.parameter_values()

    x_min, x_max = gen.x_min, gen.x_max
    experiment = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)

    injected = {}
    for name, (lo, hi) in true_signal.bounds.items():
        lo_f, hi_f = float(lo), float(hi)
        if hi_f > lo_f:
            injected[name] = (lo_f, hi_f)

    locator = SequentialBayesianExperimentDesignLocator.create(
        builder=one_peak_lorentzian_belief,
        max_steps=50,
        parameter_bounds=injected,
    )

    rng_meas = random.Random(1042)
    while not locator.done():
        x_norm = locator.next()
        obs = experiment.measure(x_norm, rng_meas)
        locator.observe(obs)

    estimates = locator.belief.estimates()

    print("Parameter comparison (true vs estimated):")
    print(f"{'param':<15} {'true':>14} {'estimated':>14} {'ratio':>8}")
    print("-" * 55)
    for name in tp:
        true_v = tp[name]
        est_v = estimates.get(name, float("nan"))
        ratio = est_v / true_v if true_v != 0 and est_v != 0 else float("nan")
        print(f"{name:<15} {true_v:>14.4e} {est_v:>14.4e} {ratio:>8.3f}")

    # Check physical amplitude = dip_depth * linewidth^2
    true_phys = tp["dip_depth"] * tp["linewidth"] ** 2
    est_phys = estimates["dip_depth"] * estimates["linewidth"] ** 2
    print("\nPhysical Amplitude (dip_depth * linewidth²):")
    print(f"  True:      {true_phys:.4e}")
    print(f"  Estimated: {est_phys:.4e}")
    print(f"  Ratio:     {est_phys / true_phys:.4f}")


if __name__ == "__main__":
    main()
