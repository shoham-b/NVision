"""Verify that generators produce signals normalized to [0, 1] in both x and y."""

import random

import numpy as np

from nvision.sim.gen.core_generators import (
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    TwoPeakCoreGenerator,
)


def verify_generator(name: str, generator, rng: random.Random) -> None:
    """Verify a generator produces normalized signals."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print("=" * 60)

    # Generate signal
    true_signal = generator.generate(rng)

    # Sample the signal across the domain
    x_values = np.linspace(generator.x_min, generator.x_max, 1000)
    y_values = [true_signal(x) for x in x_values]

    # Compute statistics
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_range = y_max - y_min

    print(f"  X domain: [{generator.x_min}, {generator.x_max}]")
    print(f"  Y range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"  Y span: {y_range:.4f}")

    # Extract peak positions
    peak_params = [p for p in true_signal.parameters if "frequency" in p.name or "position" in p.name]
    if peak_params:
        print(f"  Peak positions (x):")
        for p in peak_params:
            print(f"    - {p.name}: {p.value:.4f}")

    # Verify normalization
    x_ok = (generator.x_min == 0.0) and (generator.x_max == 1.0)
    y_ok = (0.0 <= y_min <= 0.2) and (0.8 <= y_max <= 1.1) and (y_range >= 0.5)

    if x_ok and y_ok:
        print("  ✓ Normalization: PASS (both axes in [0, 1] range)")
    else:
        if not x_ok:
            print(f"  ✗ X normalization: FAIL (not in [0, 1])")
        if not y_ok:
            print(f"  ✗ Y normalization: FAIL (range too small or outside [0, 1])")

    return x_ok and y_ok


def main():
    """Test all generator types."""
    print("\n" + "=" * 60)
    print("Signal Normalization Verification")
    print("=" * 60)
    print("\nVerifying that all generators produce signals with:")
    print("  - X coordinates in [0, 1]")
    print("  - Y values filling roughly [0, 1]")

    rng = random.Random(42)

    results = []

    # Test OnePeak Gaussian
    gen1 = OnePeakCoreGenerator(x_min=0.0, x_max=1.0, peak_type="gaussian")
    results.append(verify_generator("OnePeak Gaussian", gen1, rng))

    # Test OnePeak Lorentzian
    gen2 = OnePeakCoreGenerator(x_min=0.0, x_max=1.0, peak_type="lorentzian")
    results.append(verify_generator("OnePeak Lorentzian", gen2, rng))

    # Test TwoPeak Gaussian
    gen3 = TwoPeakCoreGenerator(x_min=0.0, x_max=1.0, peak_type_left="gaussian", peak_type_right="gaussian")
    results.append(verify_generator("TwoPeak Gaussian", gen3, rng))

    # Test NV Center Lorentzian (zero field)
    gen4 = NVCenterCoreGenerator(x_min=0.0, x_max=1.0, variant="lorentzian", zero_field=True)
    results.append(verify_generator("NV Center Lorentzian (zero field)", gen4, rng))

    # Test NV Center Lorentzian (with splitting)
    gen5 = NVCenterCoreGenerator(x_min=0.0, x_max=1.0, variant="lorentzian", zero_field=False)
    results.append(verify_generator("NV Center Lorentzian (with hyperfine)", gen5, rng))

    # Test NV Center Voigt
    gen6 = NVCenterCoreGenerator(x_min=0.0, x_max=1.0, variant="voigt", zero_field=False)
    results.append(verify_generator("NV Center Voigt", gen6, rng))

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if passed == total:
        print("  ✓ All generators produce normalized signals!")
    else:
        print(f"  ✗ {total - passed} generator(s) failed normalization")

    print()


if __name__ == "__main__":
    main()
