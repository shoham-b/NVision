"""Tests for nvision.sim.locs.refocus shape-aware dip detection and window inference.

All signals are synthetic, deterministic, and exercise multi-dip edge cases
that match the NV-center codebase.
"""

from __future__ import annotations

import numpy as np
import pytest

from nvision.sim.locs.refocus import detect_dips, infer_focus_window
from nvision.sim.locs.refocus.strategies import infer_dip_widths
from nvision.sim.locs.refocus.window import aggregate_window, infer_max_dip_width


def _gaussian_dip(x: np.ndarray, center: float, width: float, depth: float) -> np.ndarray:
    """Return a negative Gaussian dip."""
    return -depth * np.exp(-0.5 * ((x - center) / (width / 2.355)) ** 2)


class TestDetectDips:
    """Double-monotonic dip detection with shape-aware analysis."""

    def test_two_dips_monotonic_valley(self):
        """Two dips with a monotonic valley between them — Strategy B next-dip case."""
        x = np.linspace(0, 1, 200)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.35, 0.08, 0.8)
        y += _gaussian_dip(x, 0.65, 0.08, 0.8)
        noise_threshold = 0.5  # background is ~1.0, threshold at 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        assert len(dips) == 2, f"Expected 2 dips, got {len(dips)}"

        # Each dip should be within reasonable bounds of its true centre
        centers = [(lo + hi) / 2 for lo, hi in dips]
        assert 0.30 < centers[0] < 0.40
        assert 0.60 < centers[1] < 0.70

    def test_triple_dip_symmetric(self):
        """Triple-dip symmetric NV-center-like signal."""
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.30, 0.06, 0.9)
        y += _gaussian_dip(x, 0.50, 0.06, 0.9)
        y += _gaussian_dip(x, 0.70, 0.06, 0.9)
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        assert len(dips) == 3, f"Expected 3 dips, got {len(dips)}"

    def test_triple_dip_asymmetric(self):
        """Triple-dip asymmetric: left shallow but still below threshold."""
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.30, 0.06, 0.55)  # shallow left but crosses threshold
        y += _gaussian_dip(x, 0.50, 0.06, 0.90)  # deep center
        y += _gaussian_dip(x, 0.70, 0.06, 0.90)  # deep right
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        # Should still detect all three dips despite asymmetric depths
        assert len(dips) == 3, f"Expected 3 dips, got {len(dips)}"

    def test_partially_merged_dips_dense(self):
        """Two dips close enough to blur at coarse sampling; resolved when dense."""
        x = np.linspace(0, 1, 500)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.42, 0.06, 0.9)
        y += _gaussian_dip(x, 0.58, 0.06, 0.9)
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        # With dense sampling and sufficient separation, shape-aware detection
        # should resolve both dips.
        assert len(dips) == 2, f"Expected 2 dips, got {len(dips)}"

    def test_noise_only_baseline(self):
        """Flat signal with noise — no false-positive dips."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 200)
        y = np.ones_like(x) + rng.normal(0, 0.02, size=len(x))
        noise_threshold = 0.8

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        assert len(dips) == 0, f"Expected 0 dips in noise, got {len(dips)}"

    def test_missing_center_dip(self):
        """Two outer dips, center dip below noise threshold."""
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.30, 0.06, 0.9)
        # Center dip is very shallow — below threshold
        y += _gaussian_dip(x, 0.50, 0.06, 0.1)
        y += _gaussian_dip(x, 0.70, 0.06, 0.9)
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        # Should detect the two strong outer dips
        assert len(dips) == 2, f"Expected 2 dips, got {len(dips)}"
        centers = [(lo + hi) / 2 for lo, hi in dips]
        assert all(c < 0.45 or c > 0.55 for c in centers)


class TestInferDipWidths:
    """Per-dip width inference with Strategy A (background-bounded) and B (monotonic)."""

    def test_single_dip_background_bounded(self):
        """Strategy A: one dip with flat background on both sides."""
        x = np.linspace(0, 1, 200)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.5, 0.1, 0.9)
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        assert len(dips) == 1
        widths = infer_dip_widths(x, y, dips, noise_threshold=noise_threshold)
        assert len(widths) == 1
        # Width should be relatively tight (much less than the full domain)
        assert widths[0] < 0.3

    def test_two_dips_widths_bounded_by_neighbor(self):
        """Strategy B: dip widths bounded by adjacent dips."""
        x = np.linspace(0, 1, 200)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.35, 0.08, 0.8)
        y += _gaussian_dip(x, 0.65, 0.08, 0.8)
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        assert len(dips) == 2
        widths = infer_dip_widths(x, y, dips, noise_threshold=noise_threshold)
        assert len(widths) == 2
        # Each width should be less than the distance between dip centres
        assert widths[0] < 0.30  # 0.65 - 0.35 = 0.30
        assert widths[1] < 0.30

    def test_smallest_width_is_tightest_bound(self):
        """The smallest detected width is the tightest upper bound on true width."""
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.30, 0.06, 0.9)
        y += _gaussian_dip(x, 0.50, 0.06, 0.9)
        y += _gaussian_dip(x, 0.70, 0.06, 0.9)
        noise_threshold = 0.5

        dips = detect_dips(x, y, noise_threshold=noise_threshold)
        widths = infer_dip_widths(x, y, dips, noise_threshold=noise_threshold)
        assert len(widths) == 3
        max_dip_width = min(widths)
        # All dips have the same true width, so min should be close to true width
        # True FWHM ~0.06, but our inference includes some margin
        assert 0.03 < max_dip_width < 0.15


class TestInferFocusWindow:
    """Aggregate window inference (Strategy C) from per-dip widths."""

    def test_triple_dip_window_tighter_than_full_domain(self):
        """Aggregate window should be narrower than the full domain and contain all dips."""
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.30, 0.06, 0.9)
        y += _gaussian_dip(x, 0.50, 0.06, 0.9)
        y += _gaussian_dip(x, 0.70, 0.06, 0.9)

        # Build a minimal ObservationHistory-like object
        class FakeHistory:
            def __init__(self, xs, ys):
                self.xs = xs
                self.ys = ys

        history = FakeHistory(x, y)
        lo, hi = infer_focus_window(history, 0.0, 1.0, expected_dips=3, noise_threshold=0.5)

        # Should be tighter than the full [0, 1] domain
        assert hi - lo < 0.9
        # Should still contain all dips
        assert lo < 0.30
        assert hi > 0.70

    def test_fallback_when_no_dips(self):
        """When no dips detected, fail fast with ValueError instead of silently returning the full domain."""
        x = np.linspace(0, 1, 200)
        y = np.ones_like(x)

        class FakeHistory:
            def __init__(self, xs, ys):
                self.xs = xs
                self.ys = ys

        history = FakeHistory(x, y)
        with pytest.raises(ValueError, match="No dips detected"):
            infer_focus_window(history, 0.2, 0.8, expected_dips=1, noise_threshold=0.5)


class TestInferMaxDipWidth:
    """Tightest upper bound on dip width from detected dips."""

    def test_symmetric_triplet_max_width(self):
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        y += _gaussian_dip(x, 0.30, 0.06, 0.9)
        y += _gaussian_dip(x, 0.50, 0.06, 0.9)
        y += _gaussian_dip(x, 0.70, 0.06, 0.9)

        max_width = infer_max_dip_width(x, y, noise_threshold=0.5)
        assert max_width is not None
        assert 0.03 < max_width < 0.15


class TestAggregateWindow:
    """Building an overall window from dip centres and max dip width."""

    def test_triplet_aggregate(self):
        centers = [0.30, 0.50, 0.70]
        max_width = 0.10
        lo, hi = aggregate_window(centers, max_width, expected_dips=3, domain_lo=0.0, domain_hi=1.0)
        # Window should cover all three dips
        assert lo < 0.30
        assert hi > 0.70
        # Should not expand beyond domain
        assert lo >= 0.0
        assert hi <= 1.0

    def test_two_dip_aggregate(self):
        centers = [0.35, 0.65]
        max_width = 0.10
        lo, hi = aggregate_window(centers, max_width, expected_dips=2, domain_lo=0.0, domain_hi=1.0)
        assert lo < 0.35
        assert hi > 0.65
