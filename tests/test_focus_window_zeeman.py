"""Focus window must be Zeeman-symmetry aware.

The default NV model is Zeeman-split: two identical dips at
``frequency ± zeeman_split``. ``UnitCubeSMCMarginalDistribution._resample()``
computes the auto-narrowed frequency envelope from per-particle predicted dip
extents; before this fix ``zeeman_split`` was omitted from that math, so the
narrowed envelope (and the measurement x-axis, via ``sync_x``) could exclude
the actual dip locations at ``f ± Δ`` entirely.

These tests check:

1. The narrowed envelope always contains both predicted dip locations
   (containment), for both the Lorentzian and saturation-Voigt lineshapes.
2. The single-dip (no Zeeman splitting) case is unaffected (regression).
3. High-EIG acquisition candidates concentrate near *both* real dip
   locations and avoid the empty gap between them — verified by ranking
   ``expected_information_gain`` over ``get_candidates()``, not by raw
   candidate density (the envelope-wide "global grid" deliberately keeps a
   sparse, uniform background of low-value fallback candidates everywhere,
   as a hedge against a wrong belief; EIG-argmax selection is what actually
   determines where the locator measures, and that's what must avoid the
   gap).
"""

from __future__ import annotations

import numpy as np
import pytest

from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief


@pytest.fixture(autouse=True)
def _no_narrowing_delay(monkeypatch):
    monkeypatch.setenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "0")


def _concentrate_and_resample(smc, f0: float, delta0: float | None, *, seed: int, n: int):
    rng = np.random.default_rng(seed)
    f_lo, f_hi = smc.physical_param_bounds["frequency"]
    j_f = smc._param_names.index("frequency")
    smc._particles[:, j_f] = np.clip(rng.normal(loc=(f0 - f_lo) / (f_hi - f_lo), scale=1e-4, size=n), 0.0, 1.0)
    if delta0 is not None:
        z_lo, z_hi = smc.physical_param_bounds["zeeman_split"]
        j_z = smc._param_names.index("zeeman_split")
        smc._particles[:, j_z] = np.clip(rng.normal(loc=(delta0 - z_lo) / (z_hi - z_lo), scale=1e-4, size=n), 0.0, 1.0)
    smc._resample()


class TestZeemanEnvelopeContainment:
    def test_lorentzian_envelope_contains_both_dips(self):
        smc = nv_center_smc_belief(
            num_particles=1000, with_zeeman_splitting=True, with_hyperfine_splitting=False, with_fixed_frequency=False
        )
        f_lo, f_hi = smc.physical_param_bounds["frequency"]
        f0 = 0.5 * (f_lo + f_hi)
        delta0 = 40e6
        old_lo, old_hi = f_lo, f_hi

        _concentrate_and_resample(smc, f0, delta0, seed=0, n=1000)

        new_lo, new_hi = smc.physical_param_bounds["frequency"]
        assert new_lo <= f0 - delta0, "narrowed window must reach the left dip"
        assert new_hi >= f0 + delta0, "narrowed window must reach the right dip"
        assert (new_hi - new_lo) < (old_hi - old_lo), "window must have actually narrowed"

        x_lo, x_hi = smc.physical_x_bounds
        assert x_lo <= f0 - delta0, "measurement x-axis must also reach the left dip (sync_x)"
        assert x_hi >= f0 + delta0, "measurement x-axis must also reach the right dip (sync_x)"

    def test_saturation_voigt_envelope_contains_both_dips(self):
        smc = nv_center_smc_belief(
            num_particles=1000,
            with_zeeman_splitting=True,
            with_hyperfine_splitting=False,
            with_fixed_frequency=False,
            lineshape="saturation_voigt",
        )
        f_lo, f_hi = smc.physical_param_bounds["frequency"]
        f0 = 0.5 * (f_lo + f_hi)
        delta0 = 30e6
        old_lo, old_hi = f_lo, f_hi

        _concentrate_and_resample(smc, f0, delta0, seed=1, n=1000)

        new_lo, new_hi = smc.physical_param_bounds["frequency"]
        assert new_lo <= f0 - delta0
        assert new_hi >= f0 + delta0
        assert (new_hi - new_lo) < (old_hi - old_lo)


class TestNoZeemanRegression:
    def test_single_dip_narrowing_unaffected(self):
        """Without zeeman_split in the model, narrowing behaves as before (hull around one dip)."""
        smc = nv_center_smc_belief(
            num_particles=1000, with_zeeman_splitting=False, with_hyperfine_splitting=False, with_fixed_frequency=False
        )
        assert "zeeman_split" not in smc._param_names
        f_lo, f_hi = smc.physical_param_bounds["frequency"]
        f0 = 0.5 * (f_lo + f_hi)
        old_lo, old_hi = f_lo, f_hi

        _concentrate_and_resample(smc, f0, None, seed=2, n=1000)

        new_lo, new_hi = smc.physical_param_bounds["frequency"]
        assert (new_hi - new_lo) < 0.5 * (old_hi - old_lo), "should narrow well below 50% of domain"
        assert new_lo <= f0 <= new_hi, "dip centre must remain inside the window"


class TestGapAwareAcquisition:
    def test_high_eig_candidates_avoid_gap_and_cover_both_dips(self):
        smc = nv_center_smc_belief(
            num_particles=2000, with_zeeman_splitting=True, with_hyperfine_splitting=False, with_fixed_frequency=False
        )
        f_lo, f_hi = smc.physical_param_bounds["frequency"]
        f0 = 0.5 * (f_lo + f_hi)
        delta0 = 40e6  # >> linewidth, so the gap is unambiguous

        _concentrate_and_resample(smc, f0, delta0, seed=0, n=2000)

        cands = smc.get_candidates()
        eig = smc.expected_information_gain(cands, noise_std=0.02)
        order = np.argsort(eig)[::-1]

        gap_lo, gap_hi = f0 - delta0 / 3.0, f0 + delta0 / 3.0

        # Top 1% by EIG: none should fall in the empty middle third of the gap.
        top_narrow_n = max(50, len(cands) // 100)
        top_narrow = cands[order[:top_narrow_n]]
        in_gap = np.sum((top_narrow > gap_lo) & (top_narrow < gap_hi))
        assert in_gap == 0, (
            f"{in_gap}/{top_narrow_n} top-EIG candidates fall in the empty gap between dips "
            "— acquisition should never prefer measuring where no signal is expected"
        )

        # Top 5% by EIG: both real (symmetric) dip locations must be represented.
        top_wide_n = max(200, len(cands) // 20)
        top_wide = cands[order[:top_wide_n]]
        near_left = np.any(np.abs(top_wide - (f0 - delta0)) < 10e6)
        near_right = np.any(np.abs(top_wide - (f0 + delta0)) < 10e6)
        assert near_left, "expected high-EIG candidates near the left dip (f0 - zeeman_split)"
        assert near_right, "expected high-EIG candidates near the right dip (f0 + zeeman_split)"
