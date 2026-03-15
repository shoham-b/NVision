import numpy as np

from nvision.sim.locs.nv_center._jit_kernels import _expected_info_gain_jit

# Mock data
freq_grid = np.linspace(2.7e9, 3.0e9, 100)
freq_posterior = np.ones(100) / 100
n_samples = 10
current_estimates_array = np.array([5e6, 0.1, 1.0])  # linewidth, amplitude, background
noise_model_code = 0  # gaussian

print("Compiling JIT function...")
try:
    val = _expected_info_gain_jit(
        2.85e9, freq_grid, freq_posterior, n_samples, current_estimates_array, noise_model_code
    )
    print(f"Compilation success. Result: {val}")
except Exception as e:
    print(f"Compilation failed: {e}")
