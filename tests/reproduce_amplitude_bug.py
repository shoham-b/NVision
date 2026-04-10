import random

from nvision import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    NVCenterCoreGenerator,
    nv_center_belief,
)

# Reproduce the amplitude bounds mismatch
x_min = DEFAULT_NV_CENTER_FREQ_X_MIN
x_max = DEFAULT_NV_CENTER_FREQ_X_MAX
width = x_max - x_min

rng = random.Random(42)
gen = NVCenterCoreGenerator(x_min=x_min, x_max=x_max)

# Generate a true signal
true_signal = gen.generate(rng)
true_params = true_signal.parameter_values()
true_amplitude = true_params["amplitude"]

# Get default belief bounds
belief = nv_center_belief()
phys_bounds = belief.physical_param_bounds
amp_hi = phys_bounds["amplitude"][1]

print(f"Domain width: {width:.2e}")
print(f"True amplitude: {true_amplitude:.2e}")
print(f"Belief amp_hi:  {amp_hi:.2e}")

if true_amplitude > amp_hi:
    print("\nBUG CONFIRMED: True amplitude is outside belief bounds!")
    print(f"Ratio: {true_amplitude / amp_hi:.2f}")
else:
    print("\nTrue amplitude is within bounds.")

# Check linewidth as well
true_linewidth = true_params["linewidth"]
lw_hi = phys_bounds["linewidth"][1]
print(f"\nTrue linewidth: {true_linewidth:.2e}")
print(f"Belief lw_hi:   {lw_hi:.2e}")
