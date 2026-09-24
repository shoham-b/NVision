"""Tests that the belief records the ESS the resample decision was made on.

The snapshot weights cannot answer this question: ``_resample`` resets them to
uniform, so a step that resampled always reads back ``ESS == num_particles`` and
the diagnostics curve can never cross the threshold that fired it.
"""

import numpy as np
import pytest

from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.spectra.nv_center import NVCenterLorentzianModel
from tests.noise import gaussian_noise

BOUNDS = {
    "frequency": (2.7e9, 2.8e9),
    "linewidth": (1e6, 3e6),
    "split": (4e6, 6e6),
    "k_np": (1.0, 5.0),
    "c_total": (0.1, 0.9),
}


def _belief(**kwargs) -> SMCMarginalDistribution:
    # A tight noise prior makes the likelihood discriminate sharply between particles.
    return SMCMarginalDistribution(
        model=NVCenterLorentzianModel(),
        parameter_bounds=BOUNDS,
        num_particles=200,
        noise_model=gaussian_noise(1e-3, 2e-3),
        **kwargs,
    )


# An observation near the dip that most particles' predictions (0.77-0.99) miss by several noise sigmas.
SURPRISING = Observation(x=2.875e9, signal_value=0.8, noise_std=0.01)


def _ess_of_weights(smc: SMCMarginalDistribution) -> float:
    return 1.0 / float(np.sum(smc._weights**2))


def test_last_ess_is_the_pre_resample_value():
    """After an auto-resample the weights are uniform, but last_ess is sub-threshold."""
    smc = _belief(auto_resample=True)
    # This observation concentrates the weights onto a handful of particles, which is exactly
    # what drives ESS through the floor.
    smc.update(SURPRISING)

    assert smc.resampled, "a wildly surprising observation should trigger a resample"

    threshold = smc.ess_threshold * smc.num_particles

    # What the weights say after the fact: fully replenished.
    assert _ess_of_weights(smc) == pytest.approx(smc.num_particles, rel=1e-5)
    assert _ess_of_weights(smc) > threshold

    # What actually happened.
    assert smc.last_ess < threshold
    assert smc.last_ess < _ess_of_weights(smc)


def test_last_ess_recorded_without_resampling():
    """Every step records it, not only resampling ones."""
    smc = _belief(auto_resample=False)
    smc.update(Observation(x=2.75e9, signal_value=0.9, noise_std=0.5))
    assert not smc.resampled
    assert smc.last_ess == pytest.approx(_ess_of_weights(smc), rel=1e-5)


def test_last_ess_recorded_for_locator_driven_resample():
    """SBED/Sobol set auto_resample=False and call _resample() themselves."""
    smc = _belief(auto_resample=False)
    smc.update(SURPRISING)
    assert not smc.resampled
    ess_before = _ess_of_weights(smc)
    assert ess_before < smc.ess_threshold * smc.num_particles

    smc._resample()
    assert smc.resampled
    assert smc.last_ess == pytest.approx(ess_before, rel=1e-5)


def test_last_ess_survives_copy():
    """Snapshots are belief copies -- the diagnostics read it off those."""
    smc = _belief(auto_resample=True)
    smc.update(SURPRISING)
    assert smc.copy().last_ess == pytest.approx(smc.last_ess, rel=1e-5)
