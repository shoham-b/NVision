"""The drive-frequency window is the upper half [D, D + delta] of the mirror-symmetric NV spectrum."""

from __future__ import annotations

import random

import numpy as np
import pytest

from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator
from nvision.spectra.nv_center import (
    DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
    DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
    NV_DRIVE_FREQ_DELTA_HZ,
    NV_ZERO_FIELD_SPLITTING_HZ,
)


def test_window_is_upper_half_above_zero_field_splitting():
    assert DEFAULT_NV_DRIVE_FREQ_MIN_PHYS == NV_ZERO_FIELD_SPLITTING_HZ
    assert DEFAULT_NV_DRIVE_FREQ_MAX_PHYS == NV_ZERO_FIELD_SPLITTING_HZ + NV_DRIVE_FREQ_DELTA_HZ


@pytest.mark.parametrize("variant", ["lorentzian", "voigt"])
@pytest.mark.parametrize("hyperfine", ["unresolved", "n14", "n15"])
def test_generated_signal_is_centered_on_and_mirror_symmetric_about_zero_field_splitting(variant, hyperfine):
    """Symmetry is what makes the lower half redundant: S(D + d) == S(D - d) for every d."""
    signal = NVCenterCoreGenerator(variant=variant, hyperfine=hyperfine, with_zeeman_splitting=True).generate(
        random.Random(1)
    )
    assert signal.get_param_value("zero_field_freq") == NV_ZERO_FIELD_SPLITTING_HZ
    offsets = np.linspace(0.5e6, 100e6, 25)
    above = np.array([signal(NV_ZERO_FIELD_SPLITTING_HZ + d) for d in offsets])
    below = np.array([signal(NV_ZERO_FIELD_SPLITTING_HZ - d) for d in offsets])
    np.testing.assert_allclose(above, below, atol=1e-12)


def test_window_grid_dip_keeps_its_left_half_maximum_inside_the_window(monkeypatch):
    """A dip drawn near D would be cut off at the window's lower edge; the padding keeps it visible.

    The signal at D (dip plus the mirrored lower dip) must lie within half the dip depth of the baseline, so the
    left half-maximum crossing is inside ``[D, D + W]``.
    """
    from nvision.sim.presets import voigt_snr_window_param_grid_generators

    monkeypatch.setattr("nvision.spectra.nv_center.NV_ZEEMAN_EDGE_PAD_FWHM", 1.0)  # the default, whatever the env says

    generators = voigt_snr_window_param_grid_generators()
    assert generators
    for name, gen in generators:
        pad = gen.zeeman_min_phys
        assert pad > 0.0, name
        xs = np.linspace(gen.drive_freq_min_phys, gen.drive_freq_max_phys, 801)
        for seed in range(12):
            signal = gen.generate(random.Random(seed))
            assert pad <= signal.get_param_value("zeeman_shift") <= gen.zeeman_max_phys, (name, seed)
            ys = np.array([signal(x) for x in xs])
            baseline, depth = ys.max(), ys.max() - ys.min()
            assert signal(gen.drive_freq_min_phys) >= baseline - 0.5 * depth, (name, seed)


def test_window_grid_padding_that_fills_the_zeeman_range_fails_fast():
    gen = NVCenterCoreGenerator(
        variant="voigt",
        hwhm=1e6,
        sigma_inhom=0.0,
        window_fwhm=1.0,
        drive_freq_min_phys=NV_ZERO_FIELD_SPLITTING_HZ,
        drive_freq_max_phys=NV_ZERO_FIELD_SPLITTING_HZ + 2e6,
    )
    with pytest.raises(ValueError, match="widen the window"):
        _ = gen.zeeman_min_phys
