"""Test script to verify the 3-peak distance metrics (simplified version)."""

from __future__ import annotations

import math


# Inline the _pairing_error function for testing
def _pairing_error(truth: list[float], est: dict[str, float]) -> dict[str, float]:
    if len(truth) == 1:
        xh = est.get("x1_hat", est.get("x_hat"))
        err = (
            abs(float(xh) - truth[0])
            if xh is not None and isinstance(xh, int | float) and math.isfinite(float(xh))
            else math.inf
        )
        return {"abs_err_x": err}

    if len(truth) == 2:
        x1h = est.get("x1_hat")
        x2h = est.get("x2_hat")

        if (
            x1h is not None
            and isinstance(x1h, int | float)
            and math.isfinite(x1h)
            and x2h is not None
            and isinstance(x2h, int | float)
            and math.isfinite(x2h)
        ):
            xs = sorted([float(x1h), float(x2h)])
            t = sorted(truth)
            err1 = abs(xs[0] - t[0])
            err2 = abs(xs[1] - t[1])
            return {
                "abs_err_x1": err1,
                "abs_err_x2": err2,
                "pair_rmse": math.sqrt(0.5 * (err1 * err1 + err2 * err2)),
            }

        return {"abs_err_x1": math.inf, "abs_err_x2": math.inf, "pair_rmse": math.inf}

    if len(truth) == 3:
        x1h = est.get("x1_hat")
        x2h = est.get("x2_hat")
        x3h = est.get("x3_hat")

        if (
            x1h is not None
            and isinstance(x1h, int | float)
            and math.isfinite(x1h)
            and x2h is not None
            and isinstance(x2h, int | float)
            and math.isfinite(x2h)
            and x3h is not None
            and isinstance(x3h, int | float)
            and math.isfinite(x3h)
        ):
            xs = sorted([float(x1h), float(x2h), float(x3h)])
            t = sorted(truth)

            # Position errors for each peak
            err1 = abs(xs[0] - t[0])
            err2 = abs(xs[1] - t[1])
            err3 = abs(xs[2] - t[2])

            # Distance between outer peaks (1 and 3) - most relevant metric
            dist_13_true = t[2] - t[0]
            dist_13_hat = xs[2] - xs[0]
            dist_13_err = abs(dist_13_hat - dist_13_true)

            # RMSE across all three peaks
            rmse = math.sqrt((err1 * err1 + err2 * err2 + err3 * err3) / 3.0)

            result = {
                "abs_err_x1": err1,
                "abs_err_x2": err2,
                "abs_err_x3": err3,
                "dist_13_err": dist_13_err,
                "triple_rmse": rmse,
            }

            # Add split error if available in estimates
            # Split represents the distance from center to outer peaks
            split_hat = est.get("split")
            if (
                split_hat is not None
                and isinstance(split_hat, int | float)
                and math.isfinite(split_hat)
            ):
                # For 3-peak symmetric distribution, split is the distance from center to outer peaks
                # True split = (t[2] - t[1]) or (t[1] - t[0]), assuming symmetric
                split_true = (t[2] - t[1] + t[1] - t[0]) / 2.0
                result["split_err"] = abs(float(split_hat) - split_true)

            return result

        return {
            "abs_err_x1": math.inf,
            "abs_err_x2": math.inf,
            "abs_err_x3": math.inf,
            "dist_13_err": math.inf,
            "triple_rmse": math.inf,
        }

    # Fallback for unexpected number of peaks
    return {"error": math.inf}


def test_3_peak_metrics():
    """Test the 3-peak distance metrics with known values."""

    print("=" * 60)
    print("Testing 3-Peak Distance Metrics (Simplified)")
    print("=" * 60)

    # Test Case 1: Perfect estimation
    print("\nTest Case 1: Perfect Estimation")
    print("-" * 60)
    truth = [2.8e9, 2.87e9, 2.94e9]
    estimates = {
        "x1_hat": 2.8e9,
        "x2_hat": 2.87e9,
        "x3_hat": 2.94e9,
        "split": 0.07e9,
    }

    metrics = _pairing_error(truth, estimates)
    print(f"Truth peaks: {[f'{x / 1e9:.3f} GHz' for x in truth]}")
    print(
        f"Estimated peaks: {[f'{estimates[k] / 1e9:.3f} GHz' for k in ['x1_hat', 'x2_hat', 'x3_hat']]}"
    )
    print(f"\nMetrics:")
    for key, value in sorted(metrics.items()):
        if "err" in key or "rmse" in key:
            print(f"  {key}: {value / 1e6:.3f} MHz")

    # Verify all errors are zero
    assert metrics["abs_err_x1"] == 0, "Position error x1 should be 0"
    assert metrics["abs_err_x2"] == 0, "Position error x2 should be 0"
    assert metrics["abs_err_x3"] == 0, "Position error x3 should be 0"
    assert metrics["dist_13_err"] == 0, "Distance error 1-3 should be 0"
    assert metrics["triple_rmse"] == 0, "RMSE should be 0"
    assert metrics["split_err"] == 0, "Split error should be 0"
    print("\n✓ All metrics are zero as expected!")

    # Test Case 2: Outer peak distance error
    print("\n\nTest Case 2: Outer Peak Distance Error")
    print("-" * 60)
    truth = [2.8e9, 2.87e9, 2.94e9]  # Outer distance = 140 MHz
    estimates = {
        "x1_hat": 2.8e9,  # Perfect
        "x2_hat": 2.875e9,  # +5 MHz error (doesn't affect outer distance)
        "x3_hat": 2.945e9,  # +5 MHz error
        "split": 0.0725e9,  # +2.5 MHz error
    }

    metrics = _pairing_error(truth, estimates)
    print(f"Truth peaks: {[f'{x / 1e9:.3f} GHz' for x in truth]}")
    print(
        f"Estimated peaks: {[f'{estimates[k] / 1e9:.3f} GHz' for k in ['x1_hat', 'x2_hat', 'x3_hat']]}"
    )
    print(f"\nTrue outer distance (1-3): {(truth[2] - truth[0]) / 1e6:.1f} MHz")
    print(
        f"Estimated outer distance (1-3): {(estimates['x3_hat'] - estimates['x1_hat']) / 1e6:.1f} MHz"
    )
    print(f"\nMetrics:")
    for key, value in sorted(metrics.items()):
        if "err" in key or "rmse" in key:
            print(f"  {key}: {value / 1e6:.3f} MHz")

    # Outer distance: true=140MHz, estimated=145MHz -> 5MHz error
    assert abs(metrics["dist_13_err"] - 5e6) < 1e3, "Distance error 1-3 should be ~5 MHz"
    print("\n✓ Outer peak distance error is as expected!")

    # Test Case 3: Split parameter accuracy
    print("\n\nTest Case 3: Split Parameter Accuracy")
    print("-" * 60)
    truth = [2.8e9, 2.87e9, 2.94e9]  # Split = 70 MHz
    estimates = {
        "x1_hat": 2.8e9,
        "x2_hat": 2.87e9,
        "x3_hat": 2.94e9,
        "split": 0.075e9,  # 75 MHz instead of 70 MHz -> 5 MHz error
    }

    metrics = _pairing_error(truth, estimates)
    print(f"True split: {((truth[2] - truth[1] + truth[1] - truth[0]) / 2.0) / 1e6:.1f} MHz")
    print(f"Estimated split: {estimates['split'] / 1e6:.1f} MHz")
    print(f"\nMetrics:")
    for key, value in sorted(metrics.items()):
        if "err" in key or "rmse" in key:
            print(f"  {key}: {value / 1e6:.3f} MHz")

    assert abs(metrics["split_err"] - 5e6) < 1e3, "Split error should be ~5 MHz"
    print("\n✓ Split parameter error is as expected!")

    # Test Case 4: Verify no intermediate distance metrics
    print("\n\nTest Case 4: Verify Simplified Metrics")
    print("-" * 60)
    truth = [2.8e9, 2.87e9, 2.94e9]
    estimates = {"x1_hat": 2.8e9, "x2_hat": 2.87e9, "x3_hat": 2.94e9}

    metrics = _pairing_error(truth, estimates)
    print(f"Available metrics: {list(metrics.keys())}")

    # Verify intermediate distance metrics are NOT present
    assert "dist_12_err" not in metrics, "dist_12_err should not be present"
    assert "dist_23_err" not in metrics, "dist_23_err should not be present"
    # Verify essential metrics ARE present
    assert "dist_13_err" in metrics, "dist_13_err should be present"
    assert "triple_rmse" in metrics, "triple_rmse should be present"
    print("\n✓ Metrics simplified correctly - only outer distance included!")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nKey metrics for 3-peak NV center distributions:")
    print("  • abs_err_x1, abs_err_x2, abs_err_x3: Position errors")
    print("  • dist_13_err: Distance between outer peaks (most relevant)")
    print("  • split_err: Error in split parameter")
    print("  • triple_rmse: Overall RMSE across all peaks")


if __name__ == "__main__":
    test_3_peak_metrics()
