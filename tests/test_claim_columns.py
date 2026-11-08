"""Run-time columns for the article's claims (nvision/metrics/claim_columns.py) on a Lorentzian with a known answer."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from nvision.metrics.claim_columns import CLAIM_COLUMN_KEYS, compute_claim_columns


class _LorentzModel:
    """signal(x) = 1 - 1/(1 + (x - c)^2): HWHM 1 (FWHM 2), depth (contrast) 1."""

    controls_drive = False
    spec = SimpleNamespace(pack_params=lambda p: [p], unpack_params=lambda v: v[0])

    def compute_many_float64(self, xs, c):
        return 1.0 - 1.0 / (1.0 + (xs - c) ** 2)


class _GaussBelief:
    """Posterior of ``c``: Normal(mean, std); the drive window is [-10, 10]."""

    drive_freq_bounds_phys = (-10.0, 10.0)

    def __init__(self, mean: float, std: float) -> None:
        self.mean, self.std = mean, std

    def marginal_cdf(self, name, xs):
        assert name == "c"
        return np.array([0.5 * (1 + math.erf((x - self.mean) / (self.std * math.sqrt(2)))) for x in xs])


def _run(positions: list[float], beliefs: list[_GaussBelief], *, shots: int = 1, noise: float = 1.0):
    model = _LorentzModel()
    true_signal = SimpleNamespace(
        parameter_names=("c",),
        get_param_bounds=lambda name: (-5.0, 5.0),
        get_param_value=lambda name: 0.0,
        model=model,
        typed_parameters=0.0,
        model_at=lambda drive: model,
    )
    snaps = [
        SimpleNamespace(
            belief=b, obs=SimpleNamespace(drive=None, drive_freq_unit=(x + 10) / 20, n_shots=shots, noise_std=noise)
        )
        for x, b in zip(positions, beliefs, strict=True)
    ]
    return SimpleNamespace(true_signal=true_signal, snapshots=snaps)


def test_sweep_information_gives_the_speedup_of_the_article() -> None:
    """Imax / Isweep = 27 W / (8 pi Gamma) for W >> Gamma (Eq. speedup): W = 20, Gamma = 2 -> 10.74."""
    run = _run([0.0] * 4, [_GaussBelief(0.0, 1.0)] * 4)
    stage = {"stage_primary_param": "c", "stage_fisher_star": 27 / 64}  # (27/16) C^2 / Gamma^2 with C = 1, Gamma = 2
    cols = compute_claim_columns(run, stage)
    assert 27 / 64 / cols["stage_fisher_sweep"] == pytest.approx(27 * 20 / (8 * math.pi * 2), rel=0.03)


def test_pit_is_the_posterior_cdf_at_the_truth() -> None:
    run = _run([0.0] * 6, [_GaussBelief(0.0, 1.0)] * 3 + [_GaussBelief(1.0, 1.0)] * 3)
    cols = compute_claim_columns(run, {"stage_primary_param": "c", "refine_start_step": 2})
    assert cols["stage_pit_refine_start"] == pytest.approx(0.5)  # snapshot 1: centred on the truth
    assert cols["stage_pit_end"] == pytest.approx(0.1587, abs=1e-3)  # truth 1 std below the posterior mean


def test_far_share_and_far_mass_in_the_localize_stage() -> None:
    # 2 search steps, then 4 localize measurements at -8, 0.5, 8, 1 (FWHM 2 -> far means > 6 from the dip), then refine.
    positions = [0.0, 0.0, -8.0, 0.5, 8.0, 1.0, 0.0, 0.0]
    wide = _GaussBelief(0.0, 5.0)
    run = _run(positions, [wide] * 8)
    stage = {
        "stage_primary_param": "c",
        "focus_found_step": 2,
        "stage2_steps": 4,
        "refine_start_step": 7,
        "stage_fwhm_true_phys": 2.0,
    }
    cols = compute_claim_columns(run, stage)
    assert cols["stage2_far_share"] == pytest.approx(0.5)
    assert cols["stage2_far_mass"] == pytest.approx(2 * (1 - 0.5 * (1 + math.erf(1.2 / math.sqrt(2)))), rel=1e-6)


def test_columns_are_none_without_a_primary_or_a_localize_stage() -> None:
    run = _run([0.0] * 3, [_GaussBelief(0.0, 1.0)] * 3)
    assert set(compute_claim_columns(run, {"stage_primary_param": None})) == set(CLAIM_COLUMN_KEYS)
    assert all(v is None for v in compute_claim_columns(run, {"stage_primary_param": None}).values())
    cols = compute_claim_columns(run, {"stage_primary_param": "c"})
    assert cols["stage2_far_share"] is None
    assert cols["stage_pit_refine_start"] is None
    assert cols["stage_pit_end"] is not None
