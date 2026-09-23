"""compute_many_float64 must reproduce the per-point scalar ``compute`` loop exactly (sweep fit path)."""

import random

import numpy as np
import pytest

from nvision.sim.combinations import CombinationGrid

GENERATORS = [
    "NVCenter-voigt-w0.50MHz-c0.10-si0.60MHz-hfn14",
    "NVCenter-voigt-w0.50MHz-c0.10-si0.60MHz-hfunresolved",
]


@pytest.mark.parametrize("gen_name", GENERATORS)
def test_voigt_compute_many_float64_matches_scalar_loop(gen_name: str):
    combo = CombinationGrid().resolve(gen_name, "Gauss(0.01)", "SimpleSweep")
    assert combo is not None
    signal = combo.generator.generate(random.Random(3))
    model = signal.model
    params = signal.typed_parameters
    lo, hi = signal.get_param_bounds("frequency")
    xs = np.linspace(lo, hi, 65)  # shape (n_x,), physical Hz

    got = model.compute_many_float64(xs, params)
    want = np.array([float(model.compute(float(x), params)) for x in xs])

    assert got.shape == (xs.shape[0],)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


def test_base_default_matches_scalar_loop():
    """A model without a compiled override falls back to the loop in the base class."""
    from nvision.spectra.signal import SignalModel

    class _Quad(SignalModel):
        @property
        def spec(self):  # pragma: no cover - unused
            raise NotImplementedError

        def compute(self, x, params):
            return params * x * x

        def compute_vectorized_samples(self, x, samples):  # pragma: no cover - unused
            raise NotImplementedError

    xs = np.linspace(-1.0, 2.0, 7)
    got = _Quad().compute_many_float64(xs, 3.0)
    np.testing.assert_array_equal(got, 3.0 * xs * xs)
