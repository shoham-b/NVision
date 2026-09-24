"""The probe window is the upper half [D, D + delta] of the mirror-symmetric NV spectrum."""

from __future__ import annotations

import random

import numpy as np
import pytest

from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator
from nvision.spectra.nv_center import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    NV_CENTER_FREQ_DELTA_HZ,
    NV_ZERO_FIELD_SPLITTING_HZ,
)


def test_window_is_upper_half_above_zero_field_splitting():
    assert DEFAULT_NV_CENTER_FREQ_X_MIN == NV_ZERO_FIELD_SPLITTING_HZ
    assert DEFAULT_NV_CENTER_FREQ_X_MAX == NV_ZERO_FIELD_SPLITTING_HZ + NV_CENTER_FREQ_DELTA_HZ


@pytest.mark.parametrize("variant", ["lorentzian", "voigt"])
@pytest.mark.parametrize("hyperfine", ["unresolved", "n14", "n15"])
def test_generated_signal_is_centered_on_and_mirror_symmetric_about_zero_field_splitting(variant, hyperfine):
    """Symmetry is what makes the lower half redundant: S(D + d) == S(D - d) for every d."""
    signal = NVCenterCoreGenerator(variant=variant, hyperfine=hyperfine, with_zeeman_splitting=True).generate(
        random.Random(1)
    )
    assert signal.get_param_value("frequency") == NV_ZERO_FIELD_SPLITTING_HZ
    offsets = np.linspace(0.5e6, 100e6, 25)
    above = np.array([signal(NV_ZERO_FIELD_SPLITTING_HZ + d) for d in offsets])
    below = np.array([signal(NV_ZERO_FIELD_SPLITTING_HZ - d) for d in offsets])
    np.testing.assert_allclose(above, below, atol=1e-12)
