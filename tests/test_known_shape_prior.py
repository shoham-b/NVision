"""Bayesian-SBED-KnownShape: the line shape (contrast and widths) known, as the lab's Bayesian ESR runs it."""

from __future__ import annotations

import random

import pytest

from nvision.sim import presets as sim_presets
from nvision.sim.combinations import CombinationGrid
from nvision.sim.locs.bayesian.belief_builders import (
    KNOWN_PRIOR_BOUND_SIGMAS,
    KNOWN_SHAPE_STD_FRACTION,
    apply_known_prior,
    apply_known_shape,
    nv_center_smc_belief,
)


def test_apply_known_prior_narrows_the_bound_and_sets_the_prior():
    shared_priors = {"hwhm": (1.0, 0.5)}
    bounds = {"contrast_total": (0.1, 0.4), "hwhm": (0.0, 5.0), "_priors": shared_priors}
    apply_known_prior(bounds, "contrast_total", 0.25, 0.0025)
    assert bounds["contrast_total"] == pytest.approx(
        (0.25 - KNOWN_PRIOR_BOUND_SIGMAS * 0.0025, 0.25 + KNOWN_PRIOR_BOUND_SIGMAS * 0.0025)
    )
    assert bounds["_priors"]["contrast_total"] == (0.25, 0.0025)
    assert "contrast_total" not in shared_priors  # the caller's priors dict is copied, never mutated
    # The box is cut at the parameter's own bound (a width of 0 stays >= 0).
    apply_known_prior(bounds, "hwhm", 0.0, 0.1)
    assert bounds["hwhm"] == pytest.approx((0.0, KNOWN_PRIOR_BOUND_SIGMAS * 0.1))


@pytest.mark.parametrize(
    ("name", "value", "std"),
    [("missing", 1.0, 0.1), ("contrast_total", 0.5, 0.01), ("contrast_total", 0.25, 0.0)],
)
def test_apply_known_prior_fails_fast(name, value, std):
    with pytest.raises(ValueError, match=r"Known|needs a model"):
        apply_known_prior({"contrast_total": (0.1, 0.4)}, name, value, std)


def test_apply_known_shape_needs_the_contrast_and_a_width():
    bounds = {"contrast_total": (0.1, 0.4), "hwhm": (1e5, 5e6)}
    # Widths offered under both names: only the one the belief has is applied.
    apply_known_shape(bounds, {"contrast_total": (0.25, 0.0025), "hwhm": (1e6, 1e4), "hwhm_lorentz": (1e6, 1e4)})
    assert set(bounds["_priors"]) == {"contrast_total", "hwhm"}
    with pytest.raises(ValueError, match="contrast and a width"):
        apply_known_shape({"contrast_total": (0.1, 0.4)}, {"contrast_total": (0.25, 0.0025), "hwhm": (1e6, 1e4)})
    with pytest.raises(ValueError, match="not line-shape parameters"):
        apply_known_shape(bounds, {"zeeman_shift": (1e6, 1e4)})


def test_known_shape_strategy_builds_a_belief_with_the_true_shape():
    """End to end through the executor's prior and the renamed (Voigt -> Lorentzian) article-grid belief."""
    from nvision.runner.executor import _TaskRunner

    name = sim_presets.lab_mirror_generator_name(50e6, 1.82e6)
    combo = CombinationGrid().resolve(name, "SNR(5)", "Bayesian-SBED-KnownShape")
    assert combo is not None
    signal = combo.generator.generate(random.Random(0), repeat_idx=0, n_repeats=1)

    class _Experiment:
        true_signal = signal

    known = _TaskRunner._known_shape_prior_phys(_Experiment())
    truth = signal.parameter_values()
    assert known["contrast_total"] == pytest.approx((truth["contrast_total"], KNOWN_SHAPE_STD_FRACTION * 0.28))
    assert known["hwhm"][0] == pytest.approx(0.91e6)
    cfg = {k: v for k, v in combo.strategy["config"].items() if k not in ("builder", "max_steps", "known_shape")}
    belief = nv_center_smc_belief(
        signal.bounds, noise_model=combo.noise.to_noise_signal_model(), known_shape_phys=known, **cfg
    )
    bounds = belief.physical_param_bounds
    lo, hi = bounds["hwhm"]
    assert lo < 0.91e6 < hi
    assert (hi - lo) == pytest.approx(2 * KNOWN_PRIOR_BOUND_SIGMAS * 0.01 * 0.91e6)
    lo, hi = bounds["contrast_total"]
    assert lo < 0.28 < hi
    assert (hi - lo) == pytest.approx(2 * KNOWN_PRIOR_BOUND_SIGMAS * 0.01 * 0.28)
    # The line position stays free over the window.
    assert bounds["zeeman_shift"][1] == pytest.approx(combo.generator.zeeman_max_phys)


def test_plain_sbed_belief_is_unchanged_by_the_known_shape_code():
    name = "NVCenter-voigt-span20-gs0-hfunresolved"
    combo = CombinationGrid().resolve(name, "SNR(5)", "Bayesian-SBED")
    assert "known_shape" not in combo.strategy["config"]
