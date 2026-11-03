"""Held-out-sweep reference fit for real ESR recordings (nvision.metrics.lab_reference)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nvision.metrics.lab_reference import _model_signal, fit_reference, holdout_metrics
from nvision.spectra.nv_center import nv_center_lorentzian_bounds_for_domain
from nvision.tools.matlab_loader import MatlabDataFile

_TRUTH = {"zero_field_freq": 2871.5e6, "hwhm": 2.0e6, "zeeman_shift": 8.0e6, "contrast_total": 0.3}
_NOISE = 0.02


def _recording(n_shots: int = 40, seed: int = 0) -> MatlabDataFile:
    """A synthetic recording of the locator's own model plus Gaussian shot noise."""
    rng = np.random.default_rng(seed)
    drive_freq_phys = np.linspace(2830e6, 2910e6, 161)
    clean = _model_signal(drive_freq_phys, _TRUTH)
    shots = clean[:, None] + rng.normal(0.0, _NOISE, size=(drive_freq_phys.size, n_shots))
    return MatlabDataFile(
        drive_freq_phys=drive_freq_phys,
        signal=shots.mean(axis=1),
        noise_std=_NOISE,
        n_valid_shots=n_shots,
        shot_ratios=shots,
        rng=np.random.default_rng(seed + 1),
    )


def _bounds(data: MatlabDataFile) -> dict[str, tuple[float, float]]:
    """The locator's bounds as ``nv matlab`` builds them."""
    lo, hi = float(data.drive_freq_phys.min()), float(data.drive_freq_phys.max())
    bounds = nv_center_lorentzian_bounds_for_domain(lo, hi, hyperfine="unresolved", with_zeeman_splitting=True)
    bounds.update({"zero_field_freq": (lo, hi), "zeeman_shift": (0.0, (hi - lo) / 4), "contrast_total": (0.1, 1.0)})
    return bounds


def test_split_sweeps_halves_are_disjoint_and_interleaved():
    locator_half, reference_half = _recording().split_sweeps()
    used = np.isfinite(locator_half.shot_ratios).any(axis=0)
    held = np.isfinite(reference_half.shot_ratios).any(axis=0)
    assert not np.any(used & held)
    assert np.all(used | held)
    assert np.all(np.flatnonzero(used) % 2 == 0)
    assert locator_half.n_valid_shots == reference_half.n_valid_shots == 20


def test_locator_half_never_returns_a_held_out_sweep():
    data = _recording()
    locator_half, _ = data.split_sweeps()
    lo, hi = float(data.drive_freq_phys.min()), float(data.drive_freq_phys.max())
    rng = np.random.default_rng(3)
    indices = {locator_half.measure(float(u), lo, hi).sweep_index for u in rng.uniform(0, 1, 2000)}
    assert indices
    assert all(i % 2 == 0 for i in indices)


def test_split_sweeps_estimates_noise_per_half_unless_given():
    locator_half, reference_half = _recording().split_sweeps()
    assert locator_half.noise_std == pytest.approx(_NOISE, rel=0.15)
    assert reference_half.noise_std == pytest.approx(_NOISE, rel=0.15)
    fixed, _ = _recording().split_sweeps(noise_std=0.05)
    assert fixed.noise_std == 0.05


def test_split_sweeps_needs_enough_sweeps():
    with pytest.raises(ValueError, match="at least 4"):
        _recording(n_shots=3).split_sweeps()


def test_reference_fit_recovers_the_truth_within_its_sigma():
    _, reference_half = _recording().split_sweeps()
    ref = fit_reference(reference_half.drive_freq_phys, reference_half.shot_ratios, _bounds(reference_half))
    for name, truth in _TRUTH.items():
        assert abs(ref.values_phys[name] - truth) < 4 * ref.sigmas_phys[name], name
    assert 0.6 < ref.chi2_dof < 1.5
    assert ref.n_shots == 161 * 20


def test_fixed_centre_is_not_checked():
    _, reference_half = _recording().split_sweeps()
    ref = fit_reference(
        reference_half.drive_freq_phys,
        reference_half.shot_ratios,
        _bounds(reference_half),
        fixed_zero_field_freq_phys=_TRUTH["zero_field_freq"],
    )
    assert ref.values_phys["zero_field_freq"] == _TRUTH["zero_field_freq"]
    cols = holdout_metrics(ref.values_phys, {k: 1e5 for k in ref.values_phys}, ref)
    assert "holdout_z_zero_field_freq" not in cols
    assert cols["holdout_z_zeeman_shift"] == pytest.approx(0.0)


def test_holdout_z_is_signed_and_uses_the_combined_sigma():
    _, reference_half = _recording().split_sweeps()
    ref = fit_reference(reference_half.drive_freq_phys, reference_half.shot_ratios, _bounds(reference_half))
    sigma = 3 * ref.sigmas_phys["zeeman_shift"]
    combined = math.hypot(sigma, ref.sigmas_phys["zeeman_shift"])
    estimates = dict(ref.values_phys, zeeman_shift=ref.values_phys["zeeman_shift"] - 2 * combined)
    cols = holdout_metrics(estimates, {k: sigma for k in ref.values_phys}, ref)
    assert cols["holdout_z_zeeman_shift"] == pytest.approx(-2.0)
    assert cols["ref_zeeman_shift"] == ref.values_phys["zeeman_shift"]


def test_pooled_noise_is_unbiased_with_few_sweeps():
    # Two shots per bin: the median of per-bin ddof=0 stds sits ~30% low; the pooled estimate does not.
    _, reference_half = _recording(n_shots=4).split_sweeps()
    ref = fit_reference(reference_half.drive_freq_phys, reference_half.shot_ratios, _bounds(reference_half))
    assert ref.noise_std == pytest.approx(_NOISE, rel=0.15)
    assert reference_half.noise_std < 0.85 * _NOISE


def test_holdout_run_writes_its_own_columns_and_noise_name(tmp_path, monkeypatch):
    """The default (UI) path of a holdout run, written under its --dir (here tmp_path), never artifacts/."""
    import polars as pl

    from nvision.cli import matlab_cmd

    monkeypatch.setattr(MatlabDataFile, "load", classmethod(lambda cls, *a, **k: _recording(n_shots=8)))
    kwargs = dict(
        matlab_file=tmp_path / "synthetic.mat",
        run_name="src/synthetic.mat",
        directory=tmp_path,
        noise_std=None,
        max_steps=30,
        valid_shots=None,
        no_progress=True,
        no_ui=False,
        particles=500,
        infer_zero_field_freq=True,
        holdout=True,
    )
    summary = matlab_cmd._matlab_run_one(**kwargs)
    bundle = tmp_path / "matlab" / "runs" / "matlab_src_synthetic.mat_real-holdout"
    row = pl.read_parquet(bundle / "locator_results.parquet").row(0, named=True)
    assert row["generator"] == "MATLAB:src/synthetic.mat"
    assert row["noise"] == "real-holdout"
    assert row["holdout_z_zeeman_shift"] == pytest.approx(summary.holdout_z["zeeman_shift"])
    assert row["ref_chi2_dof"] > 0
    assert row["shot_reuse"] >= 1.0
    assert row["primary_converged_step"] == summary.primary_converged_step
    assert (tmp_path / "cache").is_dir()
    assert (tmp_path / "matlab" / "cache").is_dir()
    # Seeded from the run name: re-processing the file reproduces the run.
    again = matlab_cmd._matlab_run_one(**{**kwargs, "no_ui": True})
    assert again.record["estimates"] == summary.record["estimates"]


def test_shot_reuse_counts_draws_per_distinct_shot():
    locator_half, _ = _recording(n_shots=4).split_sweeps()
    lo, hi = float(locator_half.drive_freq_phys.min()), float(locator_half.drive_freq_phys.max())
    for _ in range(50):
        locator_half.measure(0.5, lo, hi)  # one bin, two even sweeps: at most 2 distinct shots
    assert locator_half.shot_reuse == pytest.approx(25.0)
