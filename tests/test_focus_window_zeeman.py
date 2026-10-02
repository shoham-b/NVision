"""Acquisition candidates must be Zeeman-symmetry aware.

The default NV model is Zeeman-split: two identical dips at
``center_freq +/- zeeman_split``. High-EIG acquisition candidates must concentrate near *both* real dip
locations and avoid the empty gap between them -- verified by ranking ``expected_information_gain``
over ``get_candidates()``, not by raw candidate density (the baseline term deliberately keeps a
sparse, uniform background of low-value candidates everywhere, as a hedge against a wrong belief;
EIG-argmax selection is what actually determines where the locator measures).

(That the *focus* contains both dips is covered in ``tests/test_focus_policy.py``.)
"""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise


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


class TestGapAwareAcquisition:
    def test_high_eig_candidates_avoid_gap_and_cover_both_dips(self):
        smc = nv_center_smc_belief(
            num_particles=2000,
            with_zeeman_splitting=True,
            hyperfine="unresolved",
            with_fixed_frequency=False,
            noise_model=gaussian_noise(),
        )
        f_lo, f_hi = smc.physical_param_bounds["frequency"]
        f0 = 0.5 * (f_lo + f_hi)
        delta0 = 40e6  # >> linewidth, so the gap is unambiguous

        _concentrate_and_resample(smc, f0, delta0, seed=0, n=2000)

        cands = smc.get_candidates()
        eig = smc.expected_information_gain(cands)
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
