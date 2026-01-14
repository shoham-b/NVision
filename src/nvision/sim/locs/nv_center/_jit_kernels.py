"""
JIT-compiled kernels for NV Center locators using Numba.
"""

import math

import numba
import numpy as np


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


@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_utility_grid_jit(freq_grid, freq_posterior, n_samples, current_estimates_array, sigma):
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
    # Sampling cannot be easily parallelized due to random state,
    # but strictly speaking random generation can be parallel if state is handled per thread.
    # For now, keep sampling sequential as it is fast compared to model evaluation.
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
    signals = np.empty((n_samples, grid_resolution), dtype=np.float64)

    # Parallelize signal generation
    for i in numba.prange(n_samples):
        f0 = f0_samples[i]
        hwhm = linewidth / 2.0
        hwhm_sq = hwhm * hwhm
        amp_hwhm_sq = amplitude * hwhm_sq

        for j in range(grid_resolution):
            freq = freq_grid[j]
            diff = freq - f0
            denom = diff * diff + hwhm_sq
            val = background - amp_hwhm_sq / denom
            signals[i, j] = val

    # Calculate variance across samples (axis 0)
    var_params = np.empty(grid_resolution, dtype=np.float64)

    # Parallelize variance calculation per frequency point
    for j in numba.prange(grid_resolution):
        # Compute variance of column j
        # Standard loop to compute variance single pass or two pass
        # Two pass for numerical stability, or just use sum/sum_sq for speed
        sum_val = 0.0
        sum_sq = 0.0
        for i in range(n_samples):
            val = signals[i, j]
            sum_val += val
            sum_sq += val * val

        mean_val = sum_val / n_samples
        # Var = E[X^2] - (E[X])^2
        var_val = (sum_sq / n_samples) - (mean_val * mean_val)

        # Ensure non-negative due to float errors
        if var_val < 0:
            var_val = 0.0

        var_params[j] = var_val

    var_noise = sigma * sigma
    utility = var_params / var_noise
    return utility


@numba.jit(nopython=True, cache=True)
def _calculate_total_log_likelihood_jit(
    measurements_x, measurements_y, params_array, noise_model_code, distribution_code
):
    """
    Calculate total log likelihood for optimization.
    params_array: [f0, linewidth, amplitude, background, gaussian_width, split, k_np]
    noise_model_code: 0 for gaussian, 1 for poisson
    distribution_code: 0 for lorentzian, 1 for voigt, 2 for voigt-zeeman
    """
    f0 = params_array[0]
    linewidth = params_array[1]
    amplitude = params_array[2]
    background = params_array[3]
    # Extra params (may be unused depending on distribution)
    gaussian_width = params_array[4]
    split = params_array[5]
    k_np = params_array[6]

    n = len(measurements_x)
    total_ll = 0.0

    sigma = 0.05
    log_const = -4.153244906  # log(2*pi*0.05^2)

    # Optimization objective is usually called sequentially by scipy.minimize
    # The number of measurements n is usually small (< 500).
    # Parallelization overhead might strictly outweigh benefits here unless n is very large.
    # Leaving sequential for now to avoid overhead in inner loop of optimizer.
    for i in range(n):
        freq = measurements_x[i]
        obs = measurements_y[i]

        pred = 0.0
        if distribution_code == 0:  # Lorentzian
            # Inline lorentzian
            hwhm = linewidth / 2.0
            diff = freq - f0
            denom = diff * diff + hwhm * hwhm
            pred = background - (amplitude * hwhm * hwhm) / denom
        elif distribution_code == 1:  # Voigt
            pred = _voigt_model(freq, f0, linewidth, gaussian_width, amplitude, background)
        elif distribution_code == 2:  # Voigt-Zeeman
            pred = _voigt_zeeman_model(freq, f0, linewidth, split, k_np, amplitude, background)

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
def _expected_info_gain_jit(  # noqa: C901
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
    # Parallel reduction for entropy
    for i in numba.prange(grid_resolution):
        p = freq_posterior[i]
        current_entropy -= p * np.log(p + 1e-300)

    # Sample true frequencies
    cdf = np.cumsum(freq_posterior)
    cdf = cdf / cdf[-1]

    true_freqs = np.empty(n_samples, dtype=np.float64)
    # Sequential sampling
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

    # Parallelize Monte Carlo samples
    for i in numba.prange(n_samples):
        true_f0 = true_freqs[i]

        # Calculate expected signal
        hwhm = linewidth / 2.0
        diff = test_frequency - true_f0
        denom = diff * diff + hwhm * hwhm
        expected_signal = background - (amplitude * hwhm * hwhm) / denom

        # Simulate measurement
        # Random generation in parallel thread is supported by Numba
        if noise_model_code == 0:  # Gaussian
            noise_std = 0.05 * abs(expected_signal) + 0.01
            sim_intensity = np.random.normal(expected_signal, noise_std)
        else:  # Poisson
            rate = max(expected_signal, 0.1)
            sim_intensity = float(np.random.poisson(rate))

        # Update posterior for this simulated measurement
        # We need to calculate likelihood over the entire grid

        # Local log_posterior array for this thread/sample
        log_posterior = np.empty(grid_resolution, dtype=np.float64)
        max_log_post = -1e300  # Init with small value

        sigma = 0.05
        log_const = -4.153244906

        for j in range(grid_resolution):
            f_grid = freq_grid[j]

            # Model at f_grid
            diff_g = test_frequency - f_grid
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
            # Note: Accessing freq_posterior purely for read is fine
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
            sample_entropy -= p * log_p

        expected_entropy_sum += sample_entropy

    expected_entropy = expected_entropy_sum / n_samples
    info_gain = current_entropy - expected_entropy
    if info_gain < 0:
        return 0.0
    return info_gain


@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_log_likelihoods_grid_jit(
    freq_grid,
    measurement_x,
    measurement_y,
    measurement_uncertainty,
    params_array,
    distribution_code,
    noise_model_code,
):
    """
    Calculate log-likelihood for a single measurement across the entire frequency grid.

    params_array: [linewidth, amplitude, background, gaussian_width, split, k_np]
    distribution_code: 0=Lorentzian, 1=Voigt, 2=Voigt-Zeeman
    noise_model_code: 0=Gaussian, 1=Poisson
    """
    n_grid = len(freq_grid)
    log_likelihoods = np.empty(n_grid, dtype=np.float64)

    linewidth = params_array[0]
    amplitude = params_array[1]
    background = params_array[2]
    # Extra params
    gaussian_width = params_array[3]
    split = params_array[4]
    k_np = params_array[5]

    # Pre-compute constants for Gaussian noise if applicable
    log_const = 0.0
    if noise_model_code == 0:
        sigma = measurement_uncertainty
        log_const = -np.log(2 * np.pi * sigma * sigma)
    else:
        sigma = 1.0  # Dummy

    obs = measurement_y
    x_val = measurement_x

    # Parallel loop over grid
    for i in numba.prange(n_grid):
        f0 = freq_grid[i]

        # Calculate prediction
        pred = 0.0
        if distribution_code == 0:  # Lorentzian
            hwhm = linewidth / 2.0
            diff = x_val - f0
            denom = diff * diff + hwhm * hwhm
            pred = background - (amplitude * hwhm * hwhm) / denom
        elif distribution_code == 1:  # Voigt
            pred = _voigt_model(x_val, f0, linewidth, gaussian_width, amplitude, background)
        elif distribution_code == 2:  # Voigt-Zeeman
            pred = _voigt_zeeman_model(x_val, f0, linewidth, split, k_np, amplitude, background)

        # Calculate Likelihood
        if noise_model_code == 0:  # Gaussian
            diff_obs = obs - pred
            term1 = diff_obs / sigma
            ll = -0.5 * term1 * term1 + 0.5 * log_const
            log_likelihoods[i] = ll
        else:  # Poisson
            safe_pred = max(pred, 1e-9)
            ll = obs * np.log(safe_pred) - safe_pred - math.lgamma(obs + 1)
            log_likelihoods[i] = ll

    return log_likelihoods


@numba.jit(nopython=True, cache=True)
def _calculate_fisher_info_jit(measurement_x, true_params_array, noise_model_code, noise_param_val):
    """
    Calculate accumulated Fisher Information.
    true_params_array: [f0, linewidth, amplitude, background]
    """
    f0 = true_params_array[0]
    linewidth = true_params_array[1]
    amplitude = true_params_array[2]
    background = true_params_array[3]

    n = len(measurement_x)
    fi_accum = np.zeros(n, dtype=np.float64)
    current_fi_sum = 0.0

    # Pre-calc for Poisson weight
    hwhm = linewidth / 2.0
    hwhm_sq = hwhm * hwhm

    # Sequential accumulation is natural, parallel accumulation is harder (scan).
    # n is usually small (steps). Leave sequential.
    for i in range(n):
        x = measurement_x[i]

        # Derivative
        diff = x - f0
        denom = diff * diff + hwhm_sq
        deriv = -amplitude * hwhm_sq * 2 * diff / (denom * denom)

        weight = 0.0
        if noise_model_code == 0:  # Gaussian
            sigma = noise_param_val
            weight = 1.0 / (sigma * sigma)
        else:  # Poisson
            mu = background - (amplitude * hwhm_sq) / denom
            if mu < 1e-9:
                mu = 1e-9
            weight = 1.0 / mu

        fi_step = deriv * deriv * weight
        current_fi_sum += fi_step
        fi_accum[i] = current_fi_sum

    return fi_accum


@numba.jit(nopython=True, cache=True, parallel=True)
def _omp_correlation_jit(y_prime, measurements_x, freq_grid, gamma):
    """
    Calculate correlations for OMP.
    y_prime: signal - background (inverted)
    """
    n_meas = len(measurements_x)
    n_grid = len(freq_grid)

    hwhm = gamma / 2.0
    hwhm_sq = hwhm * hwhm

    # Pre-allocate correlations array
    correlations = np.empty(n_grid, dtype=np.float64)

    # Parallel loop to calculate all correlations
    for j in numba.prange(n_grid):
        f_grid = freq_grid[j]

        # Dot product and norm accumulators
        dot_prod = 0.0
        norm_sq = 0.0

        for i in range(n_meas):
            mx = measurements_x[i]
            # Atom value
            diff = mx - f_grid
            denom = diff * diff + hwhm_sq
            atom_val = 1.0 / denom

            dot_prod += y_prime[i] * atom_val
            norm_sq += atom_val * atom_val

        norm = np.sqrt(norm_sq)
        if norm < 1e-9:
            norm = 1.0

        correlations[j] = dot_prod / norm

    # Find argmax (sequential or numpy's implementation is fine)
    best_idx = np.argmax(correlations)

    return freq_grid[best_idx]


@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_utility_grid_2d_jit(freq_grid, linewidth_grid, posterior_2d, n_samples, current_estimates_array, sigma):
    """
    Calculate utility for Project Bayesian Locator with 2D posterior (Freq x Linewidth).
    Returns 1D utility array over freq_grid.
    """
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)

    # Flatten posterior for sampling
    flat_posterior = posterior_2d.flatten()
    cdf = np.cumsum(flat_posterior)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    else:
        # Fallback if posterior is zero/invalid
        return np.ones(n_freq, dtype=np.float64)

    # Sample (f0, gamma) pairs
    f0_samples = np.empty(n_samples, dtype=np.float64)
    gamma_samples = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        r = np.random.random()
        idx = np.searchsorted(cdf, r)
        # Convert flat index to 2D indices
        # idx = i * n_gamma + j
        if idx >= n_freq * n_gamma:
            idx = n_freq * n_gamma - 1

        idx_f = idx // n_gamma
        idx_g = idx % n_gamma

        f0_samples[i] = freq_grid[idx_f]
        gamma_samples[i] = linewidth_grid[idx_g]

    # Unpack fixed estimates
    amplitude = current_estimates_array[1]
    background = current_estimates_array[2]

    # Calculate signals matrix: (n_samples, n_freq)
    # We evaluate utility at *each frequency in freq_grid* as a potential measurement point
    signals = np.empty((n_samples, n_freq), dtype=np.float64)

    # Parallelize signal generation
    for i in numba.prange(n_samples):
        f0 = f0_samples[i]
        gamma = gamma_samples[i]

        hwhm = gamma / 2.0
        hwhm_sq = hwhm * hwhm
        amp_hwhm_sq = amplitude * hwhm_sq

        for j in range(n_freq):
            freq = freq_grid[j]  # Test frequency
            diff = freq - f0
            denom = diff * diff + hwhm_sq
            val = background - amp_hwhm_sq / denom
            signals[i, j] = val

    # Calculate variance across samples (axis 0)
    var_params = np.empty(n_freq, dtype=np.float64)

    for j in numba.prange(n_freq):
        sum_val = 0.0
        sum_sq = 0.0
        for i in range(n_samples):
            val = signals[i, j]
            sum_val += val
            sum_sq += val * val

        mean_val = sum_val / n_samples
        var_val = (sum_sq / n_samples) - (mean_val * mean_val)

        if var_val < 0:
            var_val = 0.0
        var_params[j] = var_val

    var_noise = sigma * sigma
    utility = var_params / var_noise
    return utility


@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_log_likelihoods_2d_jit(
    freq_grid,
    linewidth_grid,
    measurement_x,
    measurement_y,
    measurement_uncertainty,
    params_array,
    distribution_code,
    noise_model_code,
):
    """
    Calculate log-likelihood on 2D grid (Freq x Linewidth).
    """
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)

    # We return a flat array or 2D? Let's return 2D.
    # Numba parallel supports multidimensional loops but requires careful indexing or prange on outer.
    log_likelihoods = np.empty((n_freq, n_gamma), dtype=np.float64)

    amplitude = params_array[1]
    background = params_array[2]
    # Extra params
    gaussian_width = params_array[3]
    split = params_array[4]
    k_np = params_array[5]

    log_const = 0.0
    if noise_model_code == 0:
        sigma = measurement_uncertainty
        log_const = -np.log(2 * np.pi * sigma * sigma)
    else:
        sigma = 1.0

    obs = measurement_y
    x_val = measurement_x

    # Flatten loop or nested? Nested is fine if we parallelize outer
    for i in numba.prange(n_freq):
        f0 = freq_grid[i]
        for j in range(n_gamma):
            gamma = linewidth_grid[j]

            # Prediction
            pred = 0.0
            if distribution_code == 0:  # Lorentzian
                hwhm = gamma / 2.0
                diff = x_val - f0
                denom = diff * diff + hwhm * hwhm
                pred = background - (amplitude * hwhm * hwhm) / denom
            elif distribution_code == 1:  # Voigt
                pred = _voigt_model(x_val, f0, gamma, gaussian_width, amplitude, background)
            elif distribution_code == 2:  # Voigt-Zeeman
                pred = _voigt_zeeman_model(x_val, f0, gamma, split, k_np, amplitude, background)

            # Likelihood
            ll = 0.0
            if noise_model_code == 0:
                diff_obs = obs - pred
                term1 = diff_obs / sigma
                ll = -0.5 * term1 * term1 + 0.5 * log_const
            else:
                safe_pred = max(pred, 1e-9)
                ll = obs * np.log(safe_pred) - safe_pred - math.lgamma(obs + 1)

            log_likelihoods[i, j] = ll

    return log_likelihoods


@numba.jit(nopython=True, cache=True)
def _update_posterior_2d_jit(freq_grid, linewidth_grid, posterior_2d, log_likelihoods_2d):
    """
    Update 2D posterior and calculate statistics.
    """
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)

    # Log posterior update
    log_prior = np.log(posterior_2d + 1e-300)
    log_posterior = log_prior + log_likelihoods_2d

    # Normalize (LogSumExp over 2D array)
    # Flatten just for simple lse
    flat_lp = log_posterior.flatten()
    lse = _logsumexp_jit(flat_lp)

    log_posterior -= lse
    new_posterior = np.exp(log_posterior)

    # Estimates
    # Marginalize for Freq
    # Sum over gamma (axis 1)
    marg_freq = np.empty(n_freq, dtype=np.float64)
    for i in range(n_freq):
        sum_p = 0.0
        for j in range(n_gamma):
            sum_p += new_posterior[i, j]
        marg_freq[i] = sum_p

    est_freq = np.sum(freq_grid * marg_freq)
    var_freq = np.sum((freq_grid - est_freq) ** 2 * marg_freq)
    uncert_freq = np.sqrt(var_freq)

    # Marginalize for Linewidth
    # Sum over freq (axis 0)
    marg_gamma = np.empty(n_gamma, dtype=np.float64)
    for j in range(n_gamma):
        sum_p = 0.0
        for i in range(n_freq):
            sum_p += new_posterior[i, j]
        marg_gamma[j] = sum_p

    est_gamma = np.sum(linewidth_grid * marg_gamma)
    var_gamma = np.sum((linewidth_grid - est_gamma) ** 2 * marg_gamma)
    uncert_gamma = np.sqrt(var_gamma)

    # Entropy
    entropy = -np.sum(new_posterior * np.log(new_posterior + 1e-300))
    max_prob = np.max(new_posterior)

    return new_posterior, marg_freq, marg_gamma, est_freq, est_gamma, uncert_freq, uncert_gamma, entropy, max_prob
