"""
JIT-compiled kernels for NV Center locators using Numba.
"""

import math
import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def _lorentzian_model(frequency, f0, linewidth, amplitude, background):
    """Calculate Lorentzian lineshape."""
    hwhm = linewidth / 2.0
    diff = frequency - f0
    denom = diff * diff + hwhm * hwhm
    return background - (amplitude * hwhm * hwhm) / denom


@numba.jit(nopython=True, cache=True)
def _pseudo_voigt(x, f0, sigma, gamma):
    """
    Calculate Pseudo-Voigt profile.
    Approximation of Voigt profile using linear combination of Gaussian and Lorentzian.
    """
    # FWHM approximations
    f_g = 2.35482 * sigma
    f_l = 2.0 * gamma
    f = (
        f_g**5
        + 2.69269 * f_g**4 * f_l
        + 2.42843 * f_g**3 * f_l**2
        + 4.47163 * f_g**2 * f_l**3
        + 0.07842 * f_g * f_l**4
        + f_l**5
    ) ** (0.2)

    eta = 1.36603 * (f_l / f) - 0.47719 * (f_l / f) ** 2 + 0.11116 * (f_l / f) ** 3

    # Gaussian part
    g_val = np.exp(-4.0 * np.log(2.0) * (x - f0) ** 2 / f**2)

    # Lorentzian part
    l_val = 1.0 / (1.0 + 4.0 * (x - f0) ** 2 / f**2)

    return eta * l_val + (1.0 - eta) * g_val


@numba.jit(nopython=True, cache=True)
def _voigt_model(frequency, f0, gamma, sigma, amplitude, background):
    """Calculate Voigt lineshape using Pseudo-Voigt approximation."""
    # Peak value normalization (approximate)
    peak_val = _pseudo_voigt(f0, f0, sigma, gamma)
    # Peak value normalization (approximate)
    peak_val = _pseudo_voigt(f0, f0, sigma, gamma)
    # Removed optimization to avoid Numba type unification issues

    profile = _pseudo_voigt(frequency, f0, sigma, gamma)

    profile = _pseudo_voigt(frequency, f0, sigma, gamma)
    return background - amplitude * profile / peak_val


@numba.jit(nopython=True, cache=True)
def _voigt_zeeman_model(frequency, f0, gamma, split, k_np, amplitude, background):
    """Calculate Voigt-Zeeman lineshape using Pseudo-Voigt approximation."""
    sigma = max(split / 10.0, 1e-9)
    sigma = max(split / 10.0, 1e-9)
    peak_val = _pseudo_voigt(f0, f0, sigma, gamma)
    # Removed optimization to avoid Numba type unification issues

    f_left = f0 - split

    f_left = f0 - split
    f_center = f0
    f_right = f0 + split

    w_left = 1.0 / max(k_np, 1e-9)
    w_center = 1.0
    w_right = max(k_np, 1e-9)

    v_left = _pseudo_voigt(frequency, f_left, sigma, gamma) / peak_val
    v_center = _pseudo_voigt(frequency, f_center, sigma, gamma) / peak_val
    v_right = _pseudo_voigt(frequency, f_right, sigma, gamma) / peak_val

    composite = w_left * v_left + w_center * v_center + w_right * v_right
    return background - amplitude * composite / max(k_np, 1e-9)


@numba.jit(nopython=True, cache=True)
def _gaussian_log_likelihood(observed, predicted, sigma, log_const):
    """Calculate Gaussian log-likelihood."""
    diff = observed - predicted
    term1 = diff / sigma
    return -0.5 * term1 * term1 - 0.5 * log_const


@numba.jit(nopython=True, cache=True)
def _poisson_log_likelihood(observed, predicted):
    """Calculate Poisson log-likelihood for array inputs."""
    safe_pred = np.maximum(predicted, 1e-9)
    n = len(observed)
    res = np.empty_like(observed, dtype=np.float64)
    for i in range(n):
        obs = observed[i]
        pred = safe_pred[i]
        res[i] = obs * np.log(pred) - pred - math.lgamma(obs + 1)
    return res


@numba.jit(nopython=True, cache=True)
def _poisson_log_likelihood_scalar(observed, predicted):
    """Calculate Poisson log-likelihood for scalar inputs."""
    safe_pred = max(predicted, 1e-9)
    return observed * np.log(safe_pred) - safe_pred - math.lgamma(observed + 1)


@numba.jit(nopython=True, cache=True)
def _poisson_log_likelihood_scalar_obs(observed_scalar, predicted_array):
    """Calculate Poisson log-likelihood for scalar observed and array predicted."""
    safe_pred = np.maximum(predicted_array, 1e-9)
    lgamma_obs = math.lgamma(observed_scalar + 1)
    return observed_scalar * np.log(safe_pred) - safe_pred - lgamma_obs


@numba.jit(nopython=True, cache=True)
def _logsumexp_jit(a):
    """JIT-compiled logsumexp for 1D array."""
    a_max = np.max(a)
    if not np.isfinite(a_max):
        a_max = 0.0

    s = 0.0
    for i in range(len(a)):
        s += np.exp(a[i] - a_max)

    return np.log(s) + a_max


@numba.jit(nopython=True, cache=True)
def _update_posterior_math(freq_grid, freq_posterior, log_likelihoods):
    """
    Perform the posterior update math:
    log_posterior = log(prior) + log_likelihood
    normalize
    calculate estimates (mean, uncertainty, entropy, max_prob)
    """
    # Log posterior update
    log_prior = np.log(freq_posterior + 1e-300)
    log_posterior = log_prior + log_likelihoods

    # Normalize
    # We implement a simple logsumexp here or use a helper
    lse = _logsumexp_jit(log_posterior)
    log_posterior -= lse
    new_posterior = np.exp(log_posterior)

    # Calculate estimates
    # Mean frequency
    est_freq = np.sum(freq_grid * new_posterior)

    # Uncertainty (std dev)
    variance = np.sum((freq_grid - est_freq) ** 2 * new_posterior)
    uncertainty = np.sqrt(variance)

    # Entropy
    entropy = -np.sum(new_posterior * np.log(new_posterior + 1e-300))

    # Max prob
    max_prob = np.max(new_posterior)

    return new_posterior, est_freq, uncertainty, entropy, max_prob


@numba.jit(nopython=True, cache=True)
def _calculate_utility_grid_jit(
    freq_grid, freq_posterior, n_samples, current_estimates_array, sigma
):
    """
    Calculate utility grid for Project Bayesian Locator.

    current_estimates_array: [linewidth, amplitude, background]
    """
    grid_resolution = len(freq_grid)

    # Sample f0 indices
    # Numba doesn't support np.random.choice with p=... fully in all versions/modes easily?
    # Actually it does in recent versions. Let's try.
    # If not, we can use inverse transform sampling.

    # Inverse transform sampling for f0
    cdf = np.cumsum(freq_posterior)
    cdf = cdf / cdf[-1]

    f0_samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        r = np.random.random()
        # Binary search for index
        idx = np.searchsorted(cdf, r)
        if idx >= grid_resolution:
            idx = grid_resolution - 1
        f0_samples[i] = freq_grid[idx]

    # Unpack estimates
    linewidth = current_estimates_array[0]
    amplitude = current_estimates_array[1]
    background = current_estimates_array[2]

    # Calculate signals matrix: (n_samples, grid_resolution)
    # We can do this with a loop to avoid large memory allocation if needed,
    # but (100, 1000) is small.

    # signals = np.empty((n_samples, grid_resolution), dtype=np.float64)
    # But we need variance across samples.

    # We can compute running variance to be even more memory efficient,
    # but let's stick to the direct approach for clarity first.

    signals = np.empty((n_samples, grid_resolution), dtype=np.float64)

    for i in range(n_samples):
        f0 = f0_samples[i]
        for j in range(grid_resolution):
            freq = freq_grid[j]
            # Inline lorentzian model
            hwhm = linewidth / 2.0
            diff = freq - f0
            denom = diff * diff + hwhm * hwhm
            val = background - (amplitude * hwhm * hwhm) / denom
            signals[i, j] = val

    # Calculate variance across samples (axis 0)
    var_params = np.empty(grid_resolution, dtype=np.float64)
    for j in range(grid_resolution):
        # Compute variance of column j
        col = signals[:, j]
        mean_val = np.mean(col)
        var_val = np.sum((col - mean_val) ** 2) / n_samples
        var_params[j] = var_val

    var_noise = sigma * sigma
    utility = var_params / var_noise
    return utility


@numba.jit(nopython=True, cache=True)
def _calculate_total_log_likelihood_jit(
    measurements_x, measurements_y, params_array, noise_model_code
):
    """
    Calculate total log likelihood for optimization.
    params_array: [f0, linewidth, amplitude, background]
    noise_model_code: 0 for gaussian, 1 for poisson
    """
    f0 = params_array[0]
    linewidth = params_array[1]
    amplitude = params_array[2]
    background = params_array[3]

    n = len(measurements_x)
    total_ll = 0.0

    sigma = 0.05
    log_const = -4.153244906  # log(2*pi*0.05^2)

    for i in range(n):
        freq = measurements_x[i]
        obs = measurements_y[i]

        # Inline lorentzian
        hwhm = linewidth / 2.0
        diff = freq - f0
        denom = diff * diff + hwhm * hwhm
        pred = background - (amplitude * hwhm * hwhm) / denom

        if noise_model_code == 0:  # Gaussian
            diff_obs = obs - pred
            term1 = diff_obs / sigma
            ll = -0.5 * term1 * term1 - 0.5 * log_const
            total_ll += ll
        else:  # Poisson
            safe_pred = max(pred, 1e-9)
            ll = obs * np.log(safe_pred) - safe_pred - math.lgamma(obs + 1)
            total_ll += ll

    return total_ll


@numba.jit(nopython=True, cache=True)
def _expected_info_gain_jit(
    test_frequency, freq_grid, freq_posterior, n_samples, current_estimates_array, noise_model_code
):
    """
    Calculate expected information gain using Monte Carlo.
    current_estimates_array: [linewidth, amplitude, background]
    noise_model_code: 0 for gaussian, 1 for poisson
    """
    grid_resolution = len(freq_grid)

    # Current entropy
    current_entropy = 0.0
    for i in range(grid_resolution):
        p = freq_posterior[i]
        current_entropy -= p * np.log(p + 1e-300)

    # Sample true frequencies
    cdf = np.cumsum(freq_posterior)
    cdf = cdf / cdf[-1]

    true_freqs = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        r = np.random.random()
        idx = np.searchsorted(cdf, r)
        if idx >= grid_resolution:
            idx = grid_resolution - 1
        true_freqs[i] = freq_grid[idx]

    linewidth = current_estimates_array[0]
    amplitude = current_estimates_array[1]
    background = current_estimates_array[2]

    expected_entropy_sum = 0.0

    for i in range(n_samples):
        true_f0 = true_freqs[i]

        # Calculate expected signal
        hwhm = linewidth / 2.0
        diff = test_frequency - true_f0
        denom = diff * diff + hwhm * hwhm
        expected_signal = background - (amplitude * hwhm * hwhm) / denom

        # Simulate measurement
        if noise_model_code == 0:  # Gaussian
            noise_std = 0.05 * abs(expected_signal) + 0.01
            sim_intensity = np.random.normal(expected_signal, noise_std)
        else:  # Poisson
            rate = max(expected_signal, 0.1)
            sim_intensity = float(np.random.poisson(rate))

        # Update posterior for this simulated measurement
        # We need to calculate likelihood over the entire grid

        # We can reuse _update_posterior_math logic but we need likelihoods first
        # Calculating likelihoods for one measurement over grid

        # Inline likelihood calculation loop over grid
        # To avoid allocating large arrays, we can compute log_posterior sum on the fly?
        # No, we need to normalize, so we need all values or at least max for logsumexp.

        # Allocate log_posterior array
        log_posterior = np.empty(grid_resolution, dtype=np.float64)
        max_log_post = -1e300  # Init with small value

        sigma = 0.05
        log_const = -4.153244906

        for j in range(grid_resolution):
            f_grid = freq_grid[j]

            # Model at f_grid
            diff_g = test_frequency - f_grid  # test_frequency is x, f_grid is param f0
            denom_g = diff_g * diff_g + hwhm * hwhm
            pred = background - (amplitude * hwhm * hwhm) / denom_g

            # Likelihood
            if noise_model_code == 0:  # Gaussian
                diff_obs = sim_intensity - pred
                term1 = diff_obs / sigma
                ll = -0.5 * term1 * term1 - 0.5 * log_const
            else:  # Poisson
                safe_pred = max(pred, 1e-9)
                ll = sim_intensity * np.log(safe_pred) - safe_pred - math.lgamma(sim_intensity + 1)

            # Prior
            log_prior = np.log(freq_posterior[j] + 1e-300)

            val = log_prior + ll
            log_posterior[j] = val
            if val > max_log_post:
                max_log_post = val

        # Normalize (LogSumExp)
        sum_exp = 0.0
        for j in range(grid_resolution):
            sum_exp += np.exp(log_posterior[j] - max_log_post)
        lse = np.log(sum_exp) + max_log_post

        # Calculate entropy of this new posterior
        sample_entropy = 0.0
        for j in range(grid_resolution):
            log_p = log_posterior[j] - lse
            p = np.exp(log_p)
            sample_entropy -= p * log_p  # log_p is log(p)

        expected_entropy_sum += sample_entropy

    expected_entropy = expected_entropy_sum / n_samples
    info_gain = current_entropy - expected_entropy
    if info_gain < 0:
        return 0.0
    return info_gain
