"""Stage boundaries (search -> focus -> refine) are read off the belief snapshots by one function."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nvision.metrics.stages import compute_stage_metrics


class _Belief:
    def __init__(self, primary_unc: float, has_dip: bool):
        self._unc = primary_unc
        self._has_dip = has_dip
        self.physical_param_bounds = {"zeeman_split": (0.0, 60e6)}

    def uncertainty(self):
        return {"zeeman_split": self._unc}

    def current_dips(self):
        return [object()] if self._has_dip else []


def _run(uncs: list[float], dip_from: int | None) -> SimpleNamespace:
    """A run whose step i has uncertainty uncs[i] and shows a dip from step index ``dip_from`` on."""
    snaps = [SimpleNamespace(belief=_Belief(u, dip_from is not None and i >= dip_from)) for i, u in enumerate(uncs)]
    true_signal = SimpleNamespace(parameter_values=lambda: {"zeeman_split": 1.0, "linewidth": 1.0})
    return SimpleNamespace(snapshots=snaps, true_signal=true_signal)


THRESH = 100e3  # NVISION_ZEEMAN_SPLIT_CONVERGENCE_THRESHOLD


def test_all_three_stages_reached():
    uncs = [10e6, 8e6, 6e6, 1e6, 500e3, 200e3, 90e3, 80e3]
    m = compute_stage_metrics(_run(uncs, dip_from=3))
    assert m["focus_found_step"] == 4  # first step whose belief shows a dip (index 3)
    assert m["max_info_step"] == 7  # first step at/after that with unc < 100 kHz (index 6)
    assert (m["stage1_steps"], m["stage2_steps"], m["stage3_steps"]) == (4, 3, 1)
    assert m["stage1_unc_start"] == 10e6
    assert m["stage1_unc_end"] == 1e6
    assert m["stage2_unc_start"] == 1e6
    assert m["stage2_unc_end"] == 90e3
    assert m["stage3_unc_start"] == 90e3
    assert m["stage3_unc_end"] == 80e3
    assert m["stage2_unc_ratio"] == pytest.approx(0.09)
    assert m["stage_primary_param"] == "zeeman_split"


def test_never_finding_a_dip_leaves_only_the_search_stage():
    m = compute_stage_metrics(_run([10e6, 9e6, 8e6], dip_from=None))
    assert m["focus_found_step"] is None
    assert m["max_info_step"] is None
    assert m["stage1_steps"] == 3
    assert m["stage2_steps"] is None
    assert m["stage3_steps"] is None


def test_converged_uncertainty_before_a_dip_does_not_count_as_max_info():
    # A tight prior can make the uncertainty small with no dip yet: stage 3 needs the focus first.
    m = compute_stage_metrics(_run([50e3, 50e3, 50e3, 50e3], dip_from=2))
    assert m["focus_found_step"] == 3
    assert m["max_info_step"] == 3  # first step at/after the focus below threshold
    assert m["stage1_steps"] == 3
    assert m["stage2_steps"] == 0
    assert m["stage3_steps"] == 1


def test_focus_without_convergence_has_no_refine_stage():
    m = compute_stage_metrics(_run([10e6, 5e6, 2e6, 1e6], dip_from=1))
    assert m["focus_found_step"] == 2
    assert m["max_info_step"] is None
    assert m["stage2_steps"] == 2
    assert m["stage3_steps"] is None


def test_empty_run_yields_no_stages():
    m = compute_stage_metrics(SimpleNamespace(snapshots=[], true_signal=None))
    assert all(v is None for v in m.values())
