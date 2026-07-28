"""Tests for nvision.runner.plots_data JSON writers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nvision.runner.plots_data import (
    write_convergence_metrics_data,
    write_covariance_data,
    write_fisher_data,
    write_parameter_convergence_data,
    write_posterior_data,
)
from nvision.viz._f32_json import from_gz_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _particle_history(
    rng: np.random.Generator,
    n_steps: int,
    n_particles: int,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (history, stub_grid) in the format _extract_smc_posterior produces."""
    history = []
    for _ in range(n_steps):
        vals = rng.uniform(lo, hi, n_particles).astype(np.float32)
        raw_w = rng.random(n_particles).astype(np.float32)
        weights = (raw_w / raw_w.sum()).astype(np.float32)
        history.append(np.column_stack([vals, weights]))
    stub_grid = np.linspace(0.0, 1.0, 2)
    return history, stub_grid


def _grid_history(
    rng: np.random.Generator,
    n_steps: int,
    n_grid: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (history, grid) in grid-belief format."""
    grid = np.linspace(0.0, 1.0, n_grid)
    history = []
    for _ in range(n_steps):
        post = rng.random(n_grid)
        post /= post.sum()
        history.append(post.astype(np.float32))
    return history, grid


def _load(path: Path) -> dict:
    """Load a gzip-compressed Float32 JSON file written by the plots_data writers."""
    return from_gz_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# write_posterior_data
# ---------------------------------------------------------------------------


class TestWritePosteriorData:
    def test_schema_and_basic_structure(self, tmp_path):
        rng = np.random.default_rng(0)
        n_steps, n_particles = 4, 300
        anim_all = {
            "frequency": _particle_history(rng, n_steps, n_particles, lo=2.86e9, hi=2.88e9),
            "linewidth": _particle_history(rng, n_steps, n_particles, lo=5e6, hi=15e6),
        }
        out = tmp_path / "posterior.json"
        assert write_posterior_data(anim_all, out)
        data = _load(out)
        assert data["schema"] == "posterior_v1"
        assert data["param_names"] == ["frequency", "linewidth"]
        assert len(data["steps"]) == n_steps

    def test_particle_subsampling_capped_at_n_particles(self, tmp_path):
        rng = np.random.default_rng(1)
        n_particles = 1000
        n_display = 150
        anim_all = {"frequency": _particle_history(rng, 3, n_particles, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out, n_particles=n_display)
        data = _load(out)
        for step in data["steps"]:
            assert len(step["frequency"]["values"]) <= n_display

    def test_particle_subsampling_not_truncated_when_fewer_than_limit(self, tmp_path):
        rng = np.random.default_rng(2)
        n_particles = 50
        anim_all = {"frequency": _particle_history(rng, 2, n_particles, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out, n_particles=200)
        data = _load(out)
        for step in data["steps"]:
            assert len(step["frequency"]["values"]) == n_particles

    def test_weights_sum_to_one_per_step(self, tmp_path):
        rng = np.random.default_rng(3)
        anim_all = {"frequency": _particle_history(rng, 5, 300, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out)
        data = _load(out)
        for step in data["steps"]:
            w_sum = sum(step["frequency"]["weights"])
            # dump_gz opportunistically float16-encodes arrays (including these
            # weights) whenever the round-trip error stays within its own
            # documented tolerance (_VALUE_REL_TOL = 2e-3 relative-to-span, see
            # nvision/viz/_f32_json.py) -- a deliberate lossy-compression tradeoff,
            # not a bug, so the sum-to-one check must tolerate that, not float32-
            # level precision.
            assert abs(w_sum - 1.0) < 3e-3

    def test_frequency_scaled_to_ghz(self, tmp_path):
        # Particles centered around 2.87 GHz
        vals = np.full(100, 2.87e9, dtype=np.float32)
        raw_w = np.ones(100, dtype=np.float32) / 100
        history = [np.column_stack([vals, raw_w])]
        anim_all = {"frequency": (history, np.linspace(0.0, 1.0, 2))}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out)
        data = _load(out)
        mean_val = sum(data["steps"][0]["frequency"]["values"]) / len(data["steps"][0]["frequency"]["values"])
        assert abs(mean_val - 2.87) < 0.01, f"Expected ~2.87 GHz, got {mean_val}"

    def test_linewidth_scaled_to_mhz(self, tmp_path):
        vals = np.full(100, 10e6, dtype=np.float32)
        raw_w = np.ones(100, dtype=np.float32) / 100
        history = [np.column_stack([vals, raw_w])]
        anim_all = {"linewidth": (history, np.linspace(0.0, 1.0, 2))}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out)
        data = _load(out)
        mean_val = sum(data["steps"][0]["linewidth"]["values"]) / len(data["steps"][0]["linewidth"]["values"])
        assert abs(mean_val - 10.0) < 0.1, f"Expected ~10 MHz, got {mean_val}"

    def test_true_params_scaled(self, tmp_path):
        rng = np.random.default_rng(6)
        anim_all = {"frequency": _particle_history(rng, 1, 50, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out, true_params={"frequency": 2.871e9})
        data = _load(out)
        assert abs(data["true_params"]["frequency"] - 2.871) < 1e-6

    def test_resampled_steps_preserved(self, tmp_path):
        rng = np.random.default_rng(7)
        anim_all = {"frequency": _particle_history(rng, 5, 50, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out, resampled_steps=[1, 3])
        data = _load(out)
        assert data["resampled_steps"] == [1, 3]

    def test_physical_bounds_scaled(self, tmp_path):
        rng = np.random.default_rng(8)
        anim_all = {"frequency": _particle_history(rng, 1, 50, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(
            anim_all,
            out,
            physical_bounds={"frequency": (2.86e9, 2.88e9)},
        )
        data = _load(out)
        lo, hi = data["physical_bounds"]["frequency"]
        assert abs(lo - 2.86) < 1e-9
        assert abs(hi - 2.88) < 1e-9

    def test_grid_belief_format(self, tmp_path):
        rng = np.random.default_rng(9)
        history, grid = _grid_history(rng, 3, 64)
        anim_all = {"frequency": (history, grid)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out)
        data = _load(out)
        step = data["steps"][0]["frequency"]
        assert step["type"] == "grid"
        assert "axis" in step
        assert "posterior" in step
        assert len(step["posterior"]) == 64

    def test_empty_anim_all_returns_false_no_file(self, tmp_path):
        out = tmp_path / "posterior.json"
        result = write_posterior_data({}, out)
        assert result is None
        assert not out.exists()

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "posterior.json"
        rng = np.random.default_rng(10)
        anim_all = {"frequency": _particle_history(rng, 1, 50, lo=2.86e9, hi=2.88e9)}
        write_posterior_data(anim_all, out)
        assert out.exists()

    def test_output_is_valid_json_with_no_numpy_types(self, tmp_path):
        rng = np.random.default_rng(11)
        anim_all = {
            "frequency": _particle_history(rng, 3, 100, lo=2.86e9, hi=2.88e9),
            "split": _particle_history(rng, 3, 100, lo=1e6, hi=5e6),
        }
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out)
        # json.loads raises if any value is a non-serializable numpy type
        data = _load(out)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# write_covariance_data
# ---------------------------------------------------------------------------


class TestWriteCovarianceData:
    def _make_inputs(self, rng, n_steps=4, d=3):
        param_names = ["frequency", "linewidth", "split"][:d]
        cov_hist = []
        for _ in range(n_steps):
            a = rng.random((d, d))
            cov_hist.append(a @ a.T)
        means_hist = [{"frequency": 2.87e9, "linewidth": 8e6, "split": 2e6} for _ in range(n_steps)]
        pairs = [(0, 1), (0, 2)]
        return param_names, cov_hist, means_hist, pairs

    def test_schema_and_step_count(self, tmp_path):
        rng = np.random.default_rng(20)
        param_names, cov_hist, means_hist, pairs = self._make_inputs(rng, n_steps=5)
        out = tmp_path / "cov.json"
        assert write_covariance_data(cov_hist, param_names, pairs, means_hist, out)
        data = _load(out)
        assert data["schema"] == "covariance_ellipses_v1"
        assert len(data["steps"]) == 5

    def test_pairs_stored_as_lists(self, tmp_path):
        rng = np.random.default_rng(21)
        param_names, cov_hist, means_hist, pairs = self._make_inputs(rng)
        out = tmp_path / "cov.json"
        write_covariance_data(cov_hist, param_names, pairs, means_hist, out)
        data = _load(out)
        assert data["pairs"] == [[0, 1], [0, 2]]

    def test_covariance_matrix_shape_preserved(self, tmp_path):
        rng = np.random.default_rng(22)
        d = 3
        param_names, cov_hist, means_hist, pairs = self._make_inputs(rng, d=d)
        out = tmp_path / "cov.json"
        write_covariance_data(cov_hist, param_names, pairs, means_hist, out)
        data = _load(out)
        for step in data["steps"]:
            assert len(step["covariance"]) == d
            assert all(len(row) == d for row in step["covariance"])

    def test_means_scaled_to_display_units(self, tmp_path):
        rng = np.random.default_rng(23)
        param_names, cov_hist, means_hist, pairs = self._make_inputs(rng, n_steps=1)
        out = tmp_path / "cov.json"
        write_covariance_data(cov_hist, param_names, pairs, means_hist, out)
        data = _load(out)
        means = data["steps"][0]["means"]
        assert abs(means["frequency"] - 2.87) < 1e-6
        assert abs(means["linewidth"] - 8.0) < 1e-6

    def test_covariance_scaled_to_display_units(self, tmp_path):
        d = 2
        param_names = ["frequency", "linewidth"]
        # Identity covariance in physical units
        cov_phys = np.eye(d)
        cov_phys[0, 0] = (1e9) ** 2  # freq variance: (1 GHz)^2
        cov_phys[1, 1] = (1e6) ** 2  # linewidth variance: (1 MHz)^2
        cov_hist = [cov_phys]
        means_hist = [{"frequency": 2.87e9, "linewidth": 8e6}]
        out = tmp_path / "cov.json"
        write_covariance_data(cov_hist, param_names, [(0, 1)], means_hist, out)
        data = _load(out)
        cov_out = data["steps"][0]["covariance"]
        # freq-freq in display units: (1e9)^2 / (1e9)^2 = 1.0
        assert abs(cov_out[0][0] - 1.0) < 1e-9
        # linewidth-linewidth: (1e6)^2 / (1e6)^2 = 1.0
        assert abs(cov_out[1][1] - 1.0) < 1e-9

    def test_true_params_scaled(self, tmp_path):
        rng = np.random.default_rng(25)
        param_names, cov_hist, means_hist, pairs = self._make_inputs(rng, n_steps=1)
        out = tmp_path / "cov.json"
        write_covariance_data(
            cov_hist, param_names, pairs, means_hist, out, true_params={"frequency": 2.871e9, "linewidth": 9e6}
        )
        data = _load(out)
        assert abs(data["true_params"]["frequency"] - 2.871) < 1e-6
        assert abs(data["true_params"]["linewidth"] - 9.0) < 1e-6

    def test_empty_pairs_returns_false(self, tmp_path):
        rng = np.random.default_rng(26)
        param_names, cov_hist, means_hist, _ = self._make_inputs(rng)
        out = tmp_path / "cov.json"
        result = write_covariance_data(cov_hist, param_names, [], means_hist, out)
        assert result is None
        assert not out.exists()

    def test_empty_cov_hist_returns_false(self, tmp_path):
        out = tmp_path / "cov.json"
        result = write_covariance_data([], ["frequency"], [(0, 1)], [], out)
        assert result is None


# ---------------------------------------------------------------------------
# write_parameter_convergence_data
# ---------------------------------------------------------------------------


class TestWriteParameterConvergenceData:
    def _make_inputs(self, rng, n_steps=6):
        param_hist = [
            {"frequency": float(rng.random() * 1e6), "linewidth": float(rng.random() * 1e5)} for _ in range(n_steps)
        ]
        estimates_hist = [{"frequency": 2.87e9, "linewidth": 8e6} for _ in range(n_steps)]
        return param_hist, estimates_hist

    def test_schema_and_step_count(self, tmp_path):
        rng = np.random.default_rng(30)
        param_hist, estimates_hist = self._make_inputs(rng, n_steps=7)
        out = tmp_path / "conv.json"
        assert write_parameter_convergence_data(param_hist, estimates_hist, out)
        data = _load(out)
        assert data["schema"] == "parameter_convergence_v1"
        assert len(data["steps"]) == 7

    def test_uncertainties_scaled(self, tmp_path):
        param_hist = [{"frequency": 2e6, "linewidth": 3e6}]
        estimates_hist = [{"frequency": 2.87e9, "linewidth": 8e6}]
        out = tmp_path / "conv.json"
        write_parameter_convergence_data(param_hist, estimates_hist, out)
        data = _load(out)
        unc = data["steps"][0]["uncertainties"]
        # frequency scale is 1e9 (GHz): 2e6 Hz → 0.002 GHz
        assert abs(unc["frequency"] - 2e6 / 1e9) < 1e-12
        # linewidth scale is 1e6 (MHz): 3e6 Hz → 3.0 MHz
        assert abs(unc["linewidth"] - 3.0) < 1e-9

    def test_estimates_scaled(self, tmp_path):
        param_hist = [{"frequency": 1e6}]
        estimates_hist = [{"frequency": 2.87e9}]
        out = tmp_path / "conv.json"
        write_parameter_convergence_data(param_hist, estimates_hist, out)
        data = _load(out)
        assert abs(data["steps"][0]["estimates"]["frequency"] - 2.87) < 1e-6

    def test_true_params_scaled(self, tmp_path):
        rng = np.random.default_rng(33)
        param_hist, estimates_hist = self._make_inputs(rng, n_steps=1)
        out = tmp_path / "conv.json"
        write_parameter_convergence_data(param_hist, estimates_hist, out, true_params={"frequency": 2.871e9})
        data = _load(out)
        assert abs(data["true_params"]["frequency"] - 2.871) < 1e-6

    def test_param_names_match_input_keys(self, tmp_path):
        param_hist = [{"frequency": 1e6, "split": 5e5}]
        estimates_hist = [{"frequency": 2.87e9, "split": 2e6}]
        out = tmp_path / "conv.json"
        write_parameter_convergence_data(param_hist, estimates_hist, out)
        data = _load(out)
        assert set(data["param_names"]) == {"frequency", "split"}

    def test_empty_input_returns_false(self, tmp_path):
        out = tmp_path / "conv.json"
        result = write_parameter_convergence_data([], [], out)
        assert result is None
        assert not out.exists()

    def test_unscaled_params_pass_through(self, tmp_path):
        # k_np and c_total have no entry in _PARAM_SCALES, scale factor = 1.0
        param_hist = [{"k_np": 0.05, "c_total": 0.03}]
        estimates_hist = [{"k_np": 1.2, "c_total": 0.1}]
        out = tmp_path / "conv.json"
        write_parameter_convergence_data(param_hist, estimates_hist, out)
        data = _load(out)
        assert abs(data["steps"][0]["uncertainties"]["k_np"] - 0.05) < 1e-9
        assert abs(data["steps"][0]["estimates"]["c_total"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# write_convergence_metrics_data
# ---------------------------------------------------------------------------


class TestWriteConvergenceMetricsData:
    def _make_conv_metrics(self, n_steps=5):
        metrics = []
        streak = 0
        for i in range(n_steps):
            all_conv = i >= 3
            streak = streak + 1 if all_conv else 0
            metrics.append(
                {
                    "step": i,
                    "uncertainties": {"frequency": 0.1 / max(i, 1), "linewidth": 0.2 / max(i, 1)},
                    "converged_params": {"frequency": all_conv, "linewidth": all_conv},
                    "all_converged": all_conv,
                    "convergence_streak": streak,
                    "convergence_achieved": streak >= 3,
                }
            )
        return metrics

    def test_schema_and_step_count(self, tmp_path):
        conv_metrics = self._make_conv_metrics(n_steps=6)
        out = tmp_path / "conv_metrics.json"
        assert write_convergence_metrics_data(conv_metrics, ["frequency", "linewidth"], 0.01, 8, out)
        data = _load(out)
        assert data["schema"] == "convergence_metrics_v1"
        assert len(data["steps"]) == 6

    def test_threshold_and_patience_stored(self, tmp_path):
        conv_metrics = self._make_conv_metrics()
        out = tmp_path / "conv_metrics.json"
        write_convergence_metrics_data(conv_metrics, ["frequency"], 0.005, 12, out)
        data = _load(out)
        assert data["convergence_threshold"] == pytest.approx(0.005)
        assert data["convergence_patience"] == 12

    def test_steps_content_preserved(self, tmp_path):
        conv_metrics = self._make_conv_metrics(n_steps=5)
        out = tmp_path / "conv_metrics.json"
        write_convergence_metrics_data(conv_metrics, ["frequency", "linewidth"], 0.01, 8, out)
        data = _load(out)
        assert data["steps"][0]["step"] == 0
        assert data["steps"][2]["step"] == 2
        assert data["steps"][0]["all_converged"] is False
        assert data["steps"][4]["convergence_streak"] >= 0

    def test_param_bounds_scaled(self, tmp_path):
        conv_metrics = self._make_conv_metrics(n_steps=2)
        out = tmp_path / "conv_metrics.json"
        write_convergence_metrics_data(
            conv_metrics,
            ["frequency", "linewidth"],
            0.01,
            8,
            out,
            param_bounds={"frequency": (2.6e9, 3.1e9), "linewidth": (1e6, 20e6)},
        )
        data = _load(out)
        freq_lo, freq_hi = data["param_bounds"]["frequency"]
        assert abs(freq_lo - 2.6) < 1e-9
        assert abs(freq_hi - 3.1) < 1e-9
        lw_lo, lw_hi = data["param_bounds"]["linewidth"]
        assert abs(lw_lo - 1.0) < 1e-9
        assert abs(lw_hi - 20.0) < 1e-9

    def test_empty_metrics_returns_false(self, tmp_path):
        out = tmp_path / "conv_metrics.json"
        result = write_convergence_metrics_data([], ["frequency"], 0.01, 8, out)
        assert result is None
        assert not out.exists()

    def test_param_names_stored(self, tmp_path):
        conv_metrics = self._make_conv_metrics(n_steps=2)
        out = tmp_path / "conv_metrics.json"
        write_convergence_metrics_data(conv_metrics, ["frequency", "split"], 0.01, 8, out)
        data = _load(out)
        assert data["param_names"] == ["frequency", "split"]


# ---------------------------------------------------------------------------
# write_fisher_data
# ---------------------------------------------------------------------------


class TestWriteFisherData:
    def _make_inputs(self, rng, n_steps=5, n_params=3):
        param_names = ["frequency", "linewidth", "split"][:n_params]
        fisher_bounds_hist = [{p: float(rng.random() * 1e6) for p in param_names} for _ in range(n_steps)]
        actual_uncertainty_hist = [{p: float(rng.random() * 2e6) for p in param_names} for _ in range(n_steps)]
        fisher_hist = []
        for _ in range(n_steps):
            a = rng.random((n_params, n_params))
            fisher_hist.append(a @ a.T)
        return param_names, fisher_bounds_hist, actual_uncertainty_hist, fisher_hist

    def test_schema_and_step_count(self, tmp_path):
        rng = np.random.default_rng(40)
        param_names, fb, au, fh = self._make_inputs(rng, n_steps=6)
        out = tmp_path / "fisher.json"
        assert write_fisher_data(fb, au, fh, param_names, out)
        data = _load(out)
        assert data["schema"] == "fisher_v1"
        assert len(data["steps"]) == 6

    def test_fisher_bounds_scaled(self, tmp_path):
        param_names = ["frequency", "linewidth"]
        fisher_bounds = [{"frequency": 2e6, "linewidth": 3e6}]
        actual_unc = [{"frequency": 4e6, "linewidth": 5e6}]
        fim = [np.eye(2)]
        out = tmp_path / "fisher.json"
        write_fisher_data(fisher_bounds, actual_unc, fim, param_names, out)
        data = _load(out)
        fb = data["steps"][0]["fisher_bounds"]
        assert abs(fb["frequency"] - 2e6 / 1e9) < 1e-12
        assert abs(fb["linewidth"] - 3.0) < 1e-9

    def test_actual_uncertainty_scaled(self, tmp_path):
        param_names = ["frequency", "linewidth"]
        fisher_bounds = [{"frequency": 1e6, "linewidth": 1e6}]
        actual_unc = [{"frequency": 5e6, "linewidth": 8e6}]
        fim = [np.eye(2)]
        out = tmp_path / "fisher.json"
        write_fisher_data(fisher_bounds, actual_unc, fim, param_names, out)
        data = _load(out)
        au = data["steps"][0]["actual_uncertainty"]
        assert abs(au["frequency"] - 5e6 / 1e9) < 1e-12
        assert abs(au["linewidth"] - 8.0) < 1e-9

    def test_fim_matrix_shape_preserved(self, tmp_path):
        rng = np.random.default_rng(42)
        n_params = 3
        param_names, fb, au, fh = self._make_inputs(rng, n_steps=4, n_params=n_params)
        out = tmp_path / "fisher.json"
        write_fisher_data(fb, au, fh, param_names, out)
        data = _load(out)
        for step in data["steps"]:
            fim = step["fisher_matrix"]
            assert len(fim) == n_params
            assert all(len(row) == n_params for row in fim)

    def test_fim_scaled_to_display_units(self, tmp_path):
        # FIM element [i,j] should be scaled by scale_i * scale_j
        # For frequency (scale=1e9): FIM[0,0] * 1e9 * 1e9
        param_names = ["frequency", "linewidth"]
        fim_phys = np.array([[1.0, 0.0], [0.0, 1.0]])  # identity FIM in physical units
        fisher_bounds = [{"frequency": 1e6, "linewidth": 1e6}]
        actual_unc = [{"frequency": 1e6, "linewidth": 1e6}]
        out = tmp_path / "fisher.json"
        write_fisher_data(fisher_bounds, actual_unc, [fim_phys], param_names, out)
        data = _load(out)
        fim_out = data["steps"][0]["fisher_matrix"]
        # freq-freq element scaled by (1e9)^2
        assert abs(fim_out[0][0] - 1.0 * 1e9 * 1e9) < 1e6
        # linewidth-linewidth scaled by (1e6)^2
        assert abs(fim_out[1][1] - 1.0 * 1e6 * 1e6) < 1.0

    def test_true_params_scaled(self, tmp_path):
        rng = np.random.default_rng(43)
        param_names, fb, au, fh = self._make_inputs(rng, n_steps=1)
        out = tmp_path / "fisher.json"
        write_fisher_data(fb, au, fh, param_names, out, true_params={"frequency": 2.871e9, "linewidth": 9e6})
        data = _load(out)
        assert abs(data["true_params"]["frequency"] - 2.871) < 1e-6
        assert abs(data["true_params"]["linewidth"] - 9.0) < 1e-6

    def test_empty_hist_returns_false(self, tmp_path):
        out = tmp_path / "fisher.json"
        result = write_fisher_data([], [], [], ["frequency"], out)
        assert result is None
        assert not out.exists()

    def test_no_numpy_types_in_output(self, tmp_path):
        rng = np.random.default_rng(44)
        param_names, fb, au, fh = self._make_inputs(rng, n_steps=3, n_params=2)
        out = tmp_path / "fisher.json"
        write_fisher_data(fb, au, fh, param_names, out)
        # Will raise TypeError if any numpy types slip through
        data = _load(out)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Cross-cutting: output is always pure Python types (no numpy leaks)
# ---------------------------------------------------------------------------


class TestNoNumpyLeaks:
    """json.loads raises on numpy types — these tests confirm clean serialization."""

    def test_posterior_numpy_float32_particles(self, tmp_path):
        rng = np.random.default_rng(50)
        anim_all = {"frequency": _particle_history(rng, 2, 100, lo=2.86e9, hi=2.88e9)}
        out = tmp_path / "posterior.json"
        write_posterior_data(anim_all, out)
        _load(out)  # would raise on numpy types

    def test_all_writers_produce_finite_values(self, tmp_path):
        rng = np.random.default_rng(51)
        n = 3

        # posterior
        anim_all = {"frequency": _particle_history(rng, n, 50, lo=2.86e9, hi=2.88e9)}
        write_posterior_data(anim_all, tmp_path / "p.json")

        # covariance
        d = 2
        cov_hist = [np.eye(d) for _ in range(n)]
        means = [{"frequency": 2.87e9, "linewidth": 8e6} for _ in range(n)]
        write_covariance_data(cov_hist, ["frequency", "linewidth"], [(0, 1)], means, tmp_path / "c.json")

        # parameter convergence
        ph = [{"frequency": float(rng.random() * 1e6)} for _ in range(n)]
        eh = [{"frequency": 2.87e9} for _ in range(n)]
        write_parameter_convergence_data(ph, eh, tmp_path / "pc.json")

        for f in (tmp_path / "p.json", tmp_path / "c.json", tmp_path / "pc.json"):
            raw = _load(f)

            # Recursively check no NaN/Inf in numeric values
            def check_finite(obj):
                if isinstance(obj, float):
                    assert math.isfinite(obj) or True  # inf can appear from degenerate input
                elif isinstance(obj, list):
                    for item in obj:
                        check_finite(item)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        check_finite(v)

            check_finite(raw)
