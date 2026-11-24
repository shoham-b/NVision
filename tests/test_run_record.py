"""The shared run record: explicit truth sources, pluggable detectors and their comparison on the same runs."""

import numpy as np
import pytest

from nvision.metrics import detection_aligned as da
from nvision.metrics.run_record import (
    DETECTORS,
    DIP_TEST,
    LAB_REFERENCE,
    SIGMA_DROP,
    SIM_TRUTH,
    RunRecord,
    compare_detectors,
)


def _record(source: str = "sim", *, drop_at: int = 20, dip_test: int | None = 18, n: int = 100) -> RunRecord:
    steps = np.arange(1, n + 1, dtype=np.float64)
    claimed = np.where(steps < drop_at, 10.0, 1.0 / np.sqrt(steps))
    return RunRecord(
        source=source,
        truth=SIM_TRUTH if source == "sim" else LAB_REFERENCE,
        steps=steps,
        estimate=None,
        claimed_sigma=claimed,
        error=claimed.copy(),
        i_max=1.0,
        window=50.0,
        fwhm=2.0,
        snr=5.0,
        budget=n,
        stop_rule="full-budget",
        known_nuisances=source == "lab",
        dip_test_step=dip_test if source == "sim" else None,
    )


def test_truth_sources_say_which_quantities_are_valid():
    assert SIM_TRUTH.ensemble_valid
    assert not LAB_REFERENCE.ensemble_valid
    assert set(DETECTORS) == {"dip-test", "sigma-drop"}


def test_sigma_drop_runs_on_simulated_runs_and_dip_test_only_there():
    r = _record()
    assert SIGMA_DROP.step(r) == 20  # the step right after the drop
    assert DIP_TEST.step(r) == 18
    with pytest.raises(ValueError, match="only for simulated runs"):
        DIP_TEST.step(_record("lab"))


def test_restart_follows_the_detector():
    r = _record()
    assert da.align(r, DIP_TEST).restart == 17  # the posterior before the measurement that fired
    assert da.align(r, SIGMA_DROP).restart == 20  # the first posterior after the drop


def test_compare_detectors_reports_the_gap():
    records = [_record(drop_at=20 + i, dip_test=18 + i) for i in range(5)] + [_record(dip_test=None)]
    out = compare_detectors(records)
    assert (out["n_runs"], out["n_both"], out["only_b"]) == (6, 5, 1)
    assert out["delta_median"] == pytest.approx(2.0)


def test_record_rejects_mismatched_arrays():
    r = _record()
    with pytest.raises(ValueError, match="equal 1-D arrays"):
        RunRecord(**{**r.__dict__, "error": r.error[:-1]})
