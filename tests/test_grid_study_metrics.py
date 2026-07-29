"""Metrics updates for the parameter-grid execution approach.

Covers: derived-quantity helpers (Jacobian propagation), derived convergence gating,
parse_gauss_sigma, milestone fc_param selection, and grid-study summary plots
(censored aggregation + convergence rates).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import polars as pl
import pytest

from nvision.sim.combinations import parse_gauss_sigma
from nvision.sim.locs.bayesian.sequential_bayesian_locator import (
    saturation_voigt_derived_bounds,
    saturation_voigt_derived_sigmas,
)
from nvision.spectra.nv_center import (
    NV_NATURAL_HWHM_HZ,
    _saturation_voigt_reparam_scalar,
    saturation_voigt_effective_hwhm_and_unc,
    saturation_voigt_realized_contrast_and_unc,
)

# ---------------------------------------------------------------------------
# Derived-quantity helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("s", "si"), [(0.02, 0.0), (1.0, 1e5), (5.0, 3e5), (30.0, 1.2e6)])
def test_effective_hwhm_matches_reparam(s, si):
    fwhm_total, _, _ = _saturation_voigt_reparam_scalar(s, si, 1.0)
    omega, _ = saturation_voigt_effective_hwhm_and_unc(s, si)
    assert omega == pytest.approx(fwhm_total / 2.0, rel=1e-9)


def test_effective_hwhm_unc_matches_finite_difference():
    s, si, sigma_s, sigma_si = 5.0, 3e5, 0.4, 2e4
    eps = 1e-6
    omega0, sigma_omega = saturation_voigt_effective_hwhm_and_unc(s, si, sigma_s, sigma_si)
    d_ds = (saturation_voigt_effective_hwhm_and_unc(s + eps, si)[0] - omega0) / eps
    d_dsi = (saturation_voigt_effective_hwhm_and_unc(s, si + eps)[0] - omega0) / eps
    expected = math.sqrt((d_ds * sigma_s) ** 2 + (d_dsi * sigma_si) ** 2)
    assert sigma_omega == pytest.approx(expected, rel=1e-4)


def test_effective_hwhm_lorentzian_limit():
    """sigma_inhom -> 0: omega collapses to the homogeneous HWHM gamma0*sqrt(1+s)."""
    omega, _ = saturation_voigt_effective_hwhm_and_unc(3.0, 0.0)
    assert omega == pytest.approx(NV_NATURAL_HWHM_HZ * math.sqrt(4.0), rel=1e-9)


def test_realized_contrast_law_and_unc():
    s, c_max, sigma_s, sigma_cm = 5.0, 0.1, 0.4, 0.01
    c, sigma_c = saturation_voigt_realized_contrast_and_unc(s, c_max, sigma_s, sigma_cm)
    assert c == pytest.approx(c_max * s / (1 + s), rel=1e-12)
    eps = 1e-7
    dc_ds = (saturation_voigt_realized_contrast_and_unc(s + eps, c_max)[0] - c) / eps
    dc_dcm = (saturation_voigt_realized_contrast_and_unc(s, c_max + eps)[0] - c) / eps
    expected = math.sqrt((dc_ds * sigma_s) ** 2 + (dc_dcm * sigma_cm) ** 2)
    assert sigma_c == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Derived convergence gating
# ---------------------------------------------------------------------------

_EST = {"frequency": 2.8e9, "saturation": 5.0, "sigma_inhom": 3e5, "c_max": 0.1, "zeeman_split": 1e7}
_BOUNDS = {
    "frequency": (2.6e9, 3.1e9),
    "saturation": (0.02, 30.0),
    "sigma_inhom": (0.0, 1.2e6),
    "c_max": (0.1, 0.4),
    "zeeman_split": (0.0, 1e8),
}


def test_derived_sigmas_present_for_saturation_voigt():
    sigmas = {"saturation": 0.4, "sigma_inhom": 2e4, "c_max": 0.01}
    derived = saturation_voigt_derived_sigmas(_EST, sigmas)
    assert derived is not None
    assert set(derived) == {"linewidth", "c_total"}
    assert derived["linewidth"] > 0
    assert derived["c_total"] > 0


def test_derived_sigmas_none_for_lorentzian_params():
    assert saturation_voigt_derived_sigmas({"frequency": 1.0, "linewidth": 1e6}, {"linewidth": 1e4}) is None
    assert saturation_voigt_derived_sigmas(_EST, {"linewidth": 1e4}) is None  # sigmas lack raw params


def test_derived_sigmas_propagate_inf_crlbs():
    sigmas = {"saturation": math.inf, "sigma_inhom": 2e4, "c_max": 0.01}
    derived = saturation_voigt_derived_sigmas(_EST, sigmas)
    assert derived is not None
    assert math.isinf(derived["linewidth"])


def test_derived_bounds_monotone_ranges():
    from nvision.spectra.nv_center import NV_SATURATION_C_MAX

    db = saturation_voigt_derived_bounds(_BOUNDS)
    lw_lo, lw_hi = db["linewidth"]
    c_lo, c_hi = db["c_total"]
    assert 0 < lw_lo < lw_hi
    # c_max is fixed (not read from _BOUNDS), so realized contrast only ever
    # approaches NV_SATURATION_C_MAX asymptotically as saturation -> infinity.
    assert 0 < c_lo < c_hi < NV_SATURATION_C_MAX
    # Effective HWHM range endpoints come from the bound corners
    assert lw_lo == pytest.approx(saturation_voigt_effective_hwhm_and_unc(0.02, 0.0)[0])
    assert lw_hi == pytest.approx(saturation_voigt_effective_hwhm_and_unc(30.0, 1.2e6)[0])


class _StubModel:
    def parameter_names(self):
        return list(_EST.keys())


class _StubBelief:
    """Minimal belief exposing what _target_params_converged touches."""

    model = _StubModel()
    physical_param_bounds = _BOUNDS

    def __init__(self, uncertainties):
        self._unc = uncertainties

    def uncertainty(self):
        return self._unc

    def estimates(self):
        return dict(_EST)


def _make_locator(uncertainties):
    from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

    loc = SequentialBayesianLocator.__new__(SequentialBayesianLocator)
    loc.belief = _StubBelief(uncertainties)
    loc._convergence_params = None
    loc.convergence_threshold = 0.01
    return loc


def test_target_params_converged_uses_derived_gating():
    # Tight frequency + tight derived quantities -> converged, even though the RAW
    # saturation uncertainty (0.25) is large in absolute s-units.
    unc = {
        "frequency": 1e4,  # << 100 kHz threshold
        "saturation": 0.25,  # raw 1%-of-bound gate would need < 0.2998 — borderline
        "sigma_inhom": 1e3,
        "c_max": 1e-4,
        "zeeman_split": 5e4,  # zeeman_split has its own absolute 100 kHz threshold (same as
        # frequency, see NVISION_ZEEMAN_SPLIT_CONVERGENCE_THRESHOLD) -- comfortably below it.
    }
    assert _make_locator(unc)._target_params_converged(unc) is True


def test_target_params_converged_fails_on_wide_derived_width():
    # Huge sigma_inhom uncertainty -> derived effective-HWHM unc dominates -> not converged.
    unc = {
        "frequency": 1e4,
        "saturation": 0.25,
        "sigma_inhom": 5e5,  # sqrt(2ln2)*5e5 ~ 590 kHz >> 1% of the ~2.1 MHz effective range
        "c_max": 1e-4,
        "zeeman_split": 5e4,
    }
    assert _make_locator(unc)._target_params_converged(unc) is False


def test_target_params_converged_lorentzian_path_unchanged():
    class _LorModel:
        def parameter_names(self):
            return ["frequency", "linewidth", "c_total"]

    from typing import ClassVar

    class _LorBelief:
        model = _LorModel()
        physical_param_bounds: ClassVar[dict[str, tuple[float, float]]] = {
            "frequency": (2.6e9, 3.1e9),
            "linewidth": (2e5, 5e6),
            "c_total": (0.1, 0.4),
        }

        def __init__(self, unc):
            self._unc = unc

        def uncertainty(self):
            return self._unc

        def estimates(self):  # must never be needed on this path
            raise AssertionError("estimates() should not be called for Lorentzian gating")

    from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

    loc = SequentialBayesianLocator.__new__(SequentialBayesianLocator)
    unc = {"frequency": 1e4, "linewidth": 1e4, "c_total": 1e-4}
    loc.belief = _LorBelief(unc)
    loc._convergence_params = None
    loc.convergence_threshold = 0.01
    assert loc._target_params_converged(unc) is True


# ---------------------------------------------------------------------------
# parse_gauss_sigma
# ---------------------------------------------------------------------------


def test_parse_gauss_sigma_roundtrips_preset_names():
    from nvision.sim import presets

    for name, _ in presets.sbed_study_noises():
        sigma = parse_gauss_sigma(name)
        assert sigma is not None
        assert name == f"Gauss({sigma})"


def test_parse_gauss_sigma_non_gauss():
    assert parse_gauss_sigma("NoNoise") is None
    assert parse_gauss_sigma("Poisson(5000)") is None


# ---------------------------------------------------------------------------
# Milestone fc_param selection
# ---------------------------------------------------------------------------


def test_default_fc_param_prefers_zeeman_split():
    from nvision.metrics.milestones import default_fc_param

    class _Sig:
        def __init__(self, params):
            self._p = params

        def parameter_values(self):
            return self._p

    class _RR:
        def __init__(self, params):
            self.true_signal = _Sig(params)

    assert default_fc_param(_RR({"frequency": 1.0, "zeeman_split": 2.0})) == "zeeman_split"
    assert default_fc_param(_RR({"frequency": 1.0, "split": 2.0})) == "split"
    assert default_fc_param(_RR({"frequency": 1.0})) == "split"


# ---------------------------------------------------------------------------
# Grid-study summary plots (censoring + rates)
# ---------------------------------------------------------------------------


def _grid_rows():
    rows = []
    for s in (1.0, 10.0):
        for si in (0.0, 5e5):
            for ns in (0.02, 0.06):
                for rep in range(4):
                    # One cell systematically infeasible; one repeat elsewhere unconverged.
                    infeasible_cell = s == 1.0 and si == 0.0 and ns == 0.06
                    lone_failure = s == 10.0 and si == 5e5 and ns == 0.02 and rep == 0
                    converged = not infeasible_cell and not lone_failure
                    rows.append(
                        {
                            "generator": f"NVCenter-saturation_voigt-s{s:.2f}-si{si / 1e6:.2f}MHz",
                            "noise": f"Gauss({ns})",
                            "strategy": "Bayesian-SBED",
                            "grid_saturation": s,
                            "grid_sigma_inhom": si,
                            "noise_sigma": ns,
                            "freq_converged_step": (40 + rep * 10) if converged else None,
                            "failure_reason": None
                            if converged
                            else ("infeasible_crlb" if infeasible_cell else "max_steps"),
                        }
                    )
    return rows


def _load_gz_entry(viz, entry: dict) -> dict:
    """Decompress a plot entry's payload from Viz's in-memory store.

    Viz._emit() never touches disk (see nvision/viz/base.py) -- the gzip bytes
    for a manifest entry's "path" live in ``viz.collected_bytes`` for an API
    endpoint to serve on demand, so tests read from there instead of the
    filesystem.
    """
    return json.loads(gzip.decompress(viz.collected_bytes[entry["path"]]))


def test_grid_study_heatmap_censoring_and_rates(tmp_path: Path):
    from nvision.viz import Viz

    df = pl.DataFrame(_grid_rows())
    viz = Viz(tmp_path)
    entries = viz.plot_grid_study(df)

    heatmaps = [e for e in entries if e["kind"] == "heatmap"]
    assert len(heatmaps) == 2  # one per noise level
    assert all(e["type"] == "grid_summary" for e in entries)

    by_noise = {e["noise"]: e for e in heatmaps}
    hi_noise = _load_gz_entry(viz, by_noise["0.06"])
    # Infeasible cell: x=1.0 (index 0), y=0.0 (index 0) -> censored median, rate 0, 4 infeasible
    assert hi_noise["z"][0][0] is None
    assert hi_noise["convergence_rate"][0][0] == 0.0
    assert hi_noise["n_infeasible"][0][0] == 4

    lo_noise = _load_gz_entry(viz, by_noise["0.02"])
    # Lone-failure cell: x=10.0 (index 1), y=5e5 (index 1) -> 3/4 converged, median over converged only
    assert lo_noise["convergence_rate"][1][1] == pytest.approx(0.75)
    assert lo_noise["z"][1][1] == pytest.approx(60.0)  # median of 50, 60, 70
    assert lo_noise["n_infeasible"][1][1] == 0


def test_grid_study_vs_noise_series(tmp_path: Path):
    from nvision.viz import Viz

    df = pl.DataFrame(_grid_rows())
    viz = Viz(tmp_path)
    entries = viz.plot_grid_study(df)
    vs = [e for e in entries if e["kind"] == "vs_noise"]
    assert len(vs) == 1
    payload = _load_gz_entry(viz, vs[0])
    assert payload["_graph_type"] == "chart"
    assert len(payload["series"]) == 4  # one per grid point
    for series in payload["series"]:
        assert series["x"] == [0.02, 0.06]
        assert len(series["convergence_rate"]) == 2


def test_grid_study_noop_without_grid_columns(tmp_path: Path):
    from nvision.viz import Viz

    df = pl.DataFrame(
        {
            "generator": ["NVCenter-lorentzian"],
            "noise": ["Gauss(0.01)"],
            "strategy": ["SimpleSweep"],
            "freq_converged_step": [10],
        }
    )
    assert Viz(tmp_path).plot_grid_study(df) == []
