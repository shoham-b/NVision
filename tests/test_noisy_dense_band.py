"""The Polars-free Gaussian noisy-band path must match the generic per-draw path bit-for-bit."""

from __future__ import annotations

import random

import numpy as np
import pytest

from nvision.models.noise import CompositeOverFrequencyNoise
from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise
from nvision.sim.batch import OverFrequencyNoise
from nvision.viz import measurements as m


def _reference_band(xs, ys, noise, noise_scale, n_draws, seed):
    """The pre-optimisation implementation: one DataBatch/Polars round-trip per draw."""
    seed_rng = random.Random(seed)
    draws = np.empty((n_draws, len(xs)), dtype=float)
    for i in range(n_draws):
        draws[i] = m._compute_noisy_dense_values(
            xs, ys, noise, noise_scale, rng=random.Random(seed_rng.randint(0, 2**31 - 1))
        )
    return tuple(np.percentile(draws, q, axis=0) for q in (15.87, 84.13, 2.28, 97.72))


def _ys(n: int) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n)
    return 1.0 - 0.3 / (1.0 + ((xs - 0.4) / 0.02) ** 2)


@pytest.mark.parametrize(
    ("parts", "noise_scale"),
    [
        ([OverFrequencyGaussianNoise(std=0.05)], 1.0),
        ([OverFrequencyGaussianNoise(std=0.05)], 0.7),
        (
            [OverFrequencyGaussianNoise(std=0.02), OverFrequencyGaussianNoise(std=0.03, clip_min=0.9, clip_max=1.05)],
            1.0,
        ),
        ([OverFrequencyGaussianNoise(std=0.0)], 1.0),
    ],
)
def test_gaussian_band_matches_generic_path_exactly(parts, noise_scale):
    xs = np.linspace(0.0, 1.0, 600)
    ys = _ys(600)
    noise = CompositeOverFrequencyNoise(parts)
    expected = _reference_band(xs, ys, noise, noise_scale, 40, 3)
    got = m._compute_noisy_dense_band(xs, ys, noise, noise_scale, n_draws=40, seed=3)
    for e, g in zip(expected, got, strict=True):
        np.testing.assert_array_equal(g, e)


def test_nonfinite_dense_values_fall_back_to_clean_signal_like_generic_path():
    xs = np.linspace(0.0, 1.0, 100)
    ys = _ys(100)
    ys[10] = np.nan
    noise = CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(std=0.05)])
    expected = _reference_band(xs, ys, noise, 0.5, 20, 0)
    got = m._compute_noisy_dense_band(xs, ys, noise, 0.5, n_draws=20, seed=0)
    for e, g in zip(expected, got, strict=True):
        np.testing.assert_array_equal(g, e)


def test_non_gaussian_composite_uses_generic_path():
    class PassThrough(OverFrequencyNoise):
        def apply(self, data, rng):
            return data

        def apply_scalar(self, x, signal_value, rng):
            return signal_value

        def noise_std(self):
            return 0.0

    noise = CompositeOverFrequencyNoise([PassThrough()])
    assert m._gaussian_only_parts(noise) is None
    xs = np.linspace(0.0, 1.0, 50)
    ys = _ys(50)
    lo1, hi1, _, _ = m._compute_noisy_dense_band(xs, ys, noise, n_draws=5)
    np.testing.assert_array_equal(lo1, ys)
    np.testing.assert_array_equal(hi1, ys)
