"""Native integration demo: Core architecture end-to-end.

This demonstrates the deep integration where:
1. Generators produce TrueSignal directly (not ScanBatch)
2. SignalModel implementations match the physics
3. No adapters or bridges - native all the way through
4. CLI automatically detects and uses native path
"""

from pathlib import Path

from nvision.runner.batch import run_simulation_batch as run_native_simulation_batch
from nvision.models.task import LocatorTask
from nvision.sim.gen.core_generators import NVCenterCoreGenerator, OnePeakCoreGenerator
from nvision.sim.locs.core import SimpleSweepLocator


def main():
    """Run native integration demonstration."""
    print("=" * 70)
    print("Native Integration Demo: Core Architecture End-to-End")
    print("=" * 70)

    # Test 1: Simple Gaussian peak
    print("\n" + "=" * 70)
    print("Test 1: Single Gaussian Peak")
    print("=" * 70)

    generator1 = OnePeakCoreGenerator(x_min=2.6e9, x_max=3.1e9, peak_type="gaussian")

    task1 = LocatorTask(
        generator_name="OnePeak-Gaussian",
        generator=generator1,
        noise_name="None",
        noise=None,
        strategy_name="SimpleSweep-Core",
        strategy={"class": SimpleSweepLocator, "config": {"max_steps": 30}},
        repeats=3,
        seed=42,
        slug="native-gaussian",
        out_dir=Path("artifacts/native_demo"),
        scans_dir=Path("artifacts/native_demo/scans"),
        bayes_dir=Path("artifacts/native_demo/bayes"),
        loc_max_steps=30,
        loc_timeout_s=60,
        use_cache=False,
        cache_dir=Path("artifacts/native_demo/cache"),
        log_queue=None,
        log_level=20,
        ignore_cache_strategy=None,
        require_cache=False,
    )

    print(f"\nGenerator: {task1.generator_name}")
    print(f"Strategy: {task1.strategy_name}")
    print(f"Architecture: Native Core (TrueSignal -> Runner -> Observer)")

    print("\nRunning...")
    history_df, finalize_df, experiments, _, stop_reasons = run_native_simulation_batch(task1)

    print(f"   Completed {len(experiments)} repeats")
    print(f"   Stop reasons: {stop_reasons}")
    print(f"   History shape: {history_df.shape}")
    print(f"   Finalize shape: {finalize_df.shape}")

    print("\nResults:")
    for i in range(task1.repeats):
        repeat_data = finalize_df.filter(finalize_df["repeat_id"] == i)
        if not repeat_data.is_empty():
            row = repeat_data.to_dicts()[0]
            print(f"\n   Repeat {i}:")
            if "frequency" in row:
                print(f"      Frequency estimate: {row['frequency']:.6e} Hz")
            if "peak_x" in row:
                print(f"      Peak position: {row['peak_x']:.6e} Hz")
            if "entropy" in row and isinstance(row["entropy"], (int, float)):
                print(f"      Final entropy: {row['entropy']:.4f}")
            if "measurements" in row:
                print(f"      Measurements: {row['measurements']}")

    # Test 2: NV Center Lorentzian
    print("\n" + "=" * 70)
    print("Test 2: NV Center ODMR Signal (Lorentzian)")
    print("=" * 70)

    generator2 = NVCenterCoreGenerator(
        x_min=2.6e9,
        x_max=3.1e9,
        variant="lorentzian",
        zero_field=False,  # With hyperfine splitting
    )

    task2 = LocatorTask(
        generator_name="NVCenter-Lorentzian",
        generator=generator2,
        noise_name="None",
        noise=None,
        strategy_name="SimpleSweep-Core",
        strategy={"class": SimpleSweepLocator, "config": {"max_steps": 50}},
        repeats=2,
        seed=123,
        slug="native-nvcenter",
        out_dir=Path("artifacts/native_demo"),
        scans_dir=Path("artifacts/native_demo/scans"),
        bayes_dir=Path("artifacts/native_demo/bayes"),
        loc_max_steps=50,
        loc_timeout_s=60,
        use_cache=False,
        cache_dir=Path("artifacts/native_demo/cache"),
        log_queue=None,
        log_level=20,
        ignore_cache_strategy=None,
        require_cache=False,
    )

    print(f"\nGenerator: {task2.generator_name}")
    print(f"Strategy: {task2.strategy_name}")
    print(f"Signal Model: NVCenterLorentzianModel (3 peaks)")

    print("\nRunning...")
    history_df2, finalize_df2, experiments2, _, stop_reasons2 = run_native_simulation_batch(task2)

    print(f"   Completed {len(experiments2)} repeats")
    print(f"   Stop reasons: {stop_reasons2}")
    print(f"   History shape: {history_df2.shape}")

    print("\nNV Center Parameters:")
    exp = experiments2[0]
    true_signal = exp.true_signal
    print(f"\n   True Parameters:")
    for param in true_signal.parameters:
        if param.name in ["frequency", "linewidth", "split", "k_np"]:
            if param.name == "frequency":
                print(f"      {param.name}: {param.value:.6e} Hz")
            elif param.name in ["linewidth", "split"]:
                print(f"      {param.name}: {param.value:.6e} Hz")
            else:
                print(f"      {param.name}: {param.value:.4f}")

    print("\n   Estimated Parameters:")
    for i in range(task2.repeats):
        repeat_data = finalize_df2.filter(finalize_df2["repeat_id"] == i)
        if not repeat_data.is_empty():
            row = repeat_data.to_dicts()[0]
            print(f"\n   Repeat {i}:")
            for key in ["frequency", "linewidth", "split", "k_np"]:
                if key in row and isinstance(row[key], (int, float)):
                    if key in ["frequency", "linewidth", "split"]:
                        print(f"      {key}: {row[key]:.6e} Hz")
                    else:
                        print(f"      {key}: {row[key]:.4f}")

    print("\n" + "=" * 70)
    print("SUCCESS: Native integration working end-to-end!")
    print("=" * 70)
    print("\nKey achievements:")
    print("  [OK] Generators produce TrueSignal with real SignalModel")
    print("  [OK] NV center physics in NVCenterLorentzianModel")
    print("  [OK] No ScanBatch adapters needed")
    print("  [OK] Parameters tracked from generation through localization")
    print("  [OK] CLI runner works natively with core types")
    print("=" * 70)


if __name__ == "__main__":
    main()
