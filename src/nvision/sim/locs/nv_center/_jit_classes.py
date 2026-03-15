import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True)
def _calculate_log_likelihoods(freq_grid, linewidth_grid, mx, my, uncert, amplitude, background, noise_model_code):
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)
    log_likelihoods = np.zeros((n_freq, n_gamma), dtype=np.float32)
    hwhm_grid = linewidth_grid / 2.0
    hwhm_sq_grid = hwhm_grid * hwhm_grid

    for i in prange(n_freq):
        f = freq_grid[i]
        diff = mx - f
        diff_sq = diff * diff

        for j in range(n_gamma):
            hwhm_sq = hwhm_sq_grid[j]
            denom = diff_sq + hwhm_sq
            pred = background - (amplitude * hwhm_sq) / denom

            if noise_model_code == 0:  # Gaussian
                log_sigma_term = 1.837877 + 2.0 * np.log(uncert)
                log_const = -0.5 * log_sigma_term
                diff_obs = my - pred
                term1 = diff_obs / uncert
                ll = -0.5 * term1 * term1 + log_const
            else:  # Poisson
                safe_pred = max(pred, 1e-9)
                ll = my * np.log(safe_pred) - safe_pred

            log_likelihoods[i, j] = ll
    return log_likelihoods


@jit(nopython=True, fastmath=True, parallel=True)
def _update_jit_logic(freq_grid, linewidth_grid, posterior_2d, mx, my, uncert, amplitude, background, noise_model_code):
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)

    log_likelihoods = _calculate_log_likelihoods(
        freq_grid, linewidth_grid, mx, my, uncert, amplitude, background, noise_model_code
    )

    # Update Posterior
    # This is an array operation, usually fast enough without parallel, but can be JITted
    # log_prior = log(posterior + eps)

    # Let's do it in place or new array
    # Since we are in JIT, we can loop or use array ops
    # Parallel reduction for max and sum

    max_log_val = -1e30
    for i in range(n_freq):
        for j in range(n_gamma):
            log_val = np.log(posterior_2d[i, j] + 1e-30) + log_likelihoods[i, j]
            log_likelihoods[i, j] = log_val  # Standardize to log_posterior
            if log_val > max_log_val:
                max_log_val = log_val

    # Exponentiate and Normalize
    sum_val = 0.0
    for i in range(n_freq):
        for j in range(n_gamma):
            val = np.exp(log_likelihoods[i, j] - max_log_val)
            posterior_2d[i, j] = val
            sum_val += val

    if sum_val > 0:
        factor = 1.0 / sum_val
        for i in range(n_freq):
            for j in range(n_gamma):
                posterior_2d[i, j] *= factor


@jit(nopython=True, fastmath=True)
def _calculate_utility_jit_logic(freq_grid, linewidth_grid, posterior_2d, n_samples, amplitude, background, sigma):
    # Flatten and CDF for sampling
    flat = posterior_2d.flatten()
    cdf = np.cumsum(flat)
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    f0_samples = np.zeros(n_samples, dtype=np.float32)
    gamma_samples = np.zeros(n_samples, dtype=np.float32)

    r = np.random.random(n_samples)
    indices = np.searchsorted(cdf, r)

    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)
    max_idx = n_freq * n_gamma - 1

    for k in range(n_samples):
        idx = min(indices[k], max_idx)
        idx_f = idx // n_gamma
        idx_g = idx % n_gamma
        f0_samples[k] = freq_grid[idx_f]
        gamma_samples[k] = linewidth_grid[idx_g]

    # Calculate predictions (n_samples x n_freq)
    # and Variance over samples

    var_pred = np.zeros(n_freq, dtype=np.float32)

    # We can iterate over Frequency Grid points
    for i in range(n_freq):
        f = freq_grid[i]

        # Calculate mean over samples for this frequency
        sum_pred = 0.0
        sum_sq_pred = 0.0

        for k in range(n_samples):
            f0 = f0_samples[k]
            gamma = gamma_samples[k]

            hwhm = gamma / 2.0
            hwhm_sq = hwhm * hwhm
            diff = f - f0
            denom = diff * diff + hwhm_sq
            pred = background - (amplitude * hwhm_sq) / denom

            sum_pred += pred
            sum_sq_pred += pred * pred

        mean = sum_pred / n_samples
        mean_sq = sum_sq_pred / n_samples
        variance = mean_sq - mean * mean
        if variance < 0:
            variance = 0

        var_pred[i] = variance

    utility = var_pred / (sigma * sigma)
    return utility


@jit(nopython=True, fastmath=True)
def _calculate_marginals(freq_grid, linewidth_grid, posterior_2d):
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)
    marg_freq = np.zeros(n_freq, dtype=np.float32)
    marg_gamma = np.zeros(n_gamma, dtype=np.float32)

    for i in range(n_freq):
        sum_val = 0.0
        for j in range(n_gamma):
            sum_val += posterior_2d[i, j]
        marg_freq[i] = sum_val

    for j in range(n_gamma):
        sum_val = 0.0
        for i in range(n_freq):
            sum_val += posterior_2d[i, j]
        marg_gamma[j] = sum_val

    return marg_freq, marg_gamma


@jit(nopython=True, fastmath=True)
def _get_estimates_jit_logic(freq_grid, linewidth_grid, posterior_2d):
    n_freq = len(freq_grid)
    n_gamma = len(linewidth_grid)

    marg_freq, marg_gamma = _calculate_marginals(freq_grid, linewidth_grid, posterior_2d)

    # Stats
    est_freq = 0.0
    for i in range(n_freq):
        est_freq += freq_grid[i] * marg_freq[i]

    est_gamma = 0.0
    for j in range(n_gamma):
        est_gamma += linewidth_grid[j] * marg_gamma[j]

    var_freq = 0.0
    for i in range(n_freq):
        d = freq_grid[i] - est_freq
        var_freq += d * d * marg_freq[i]
    uncert_freq = np.sqrt(var_freq)

    var_gamma = 0.0
    for j in range(n_gamma):
        d = linewidth_grid[j] - est_gamma
        var_gamma += d * d * marg_gamma[j]
    uncert_gamma = np.sqrt(var_gamma)

    entropy = 0.0
    max_prob = 0.0
    for i in range(n_freq):
        for j in range(n_gamma):
            p = posterior_2d[i, j]
            if p > max_prob:
                max_prob = p
            if p > 0:
                entropy -= p * np.log(p)

    return est_freq, est_gamma, uncert_freq, uncert_gamma, entropy, max_prob


class Bayesian2DState:
    def __init__(self, freq_grid, linewidth_grid):
        self.freq_grid = freq_grid.astype(np.float32)
        self.linewidth_grid = linewidth_grid.astype(np.float32)
        self.grid_resolution = len(freq_grid)
        self.linewidth_resolution = len(linewidth_grid)

        self.posterior_2d = np.ones((self.grid_resolution, self.linewidth_resolution), dtype=np.float32)
        self.posterior_2d /= np.sum(self.posterior_2d)

    def update(self, mx, my, uncert, amplitude, background, noise_model_code):
        _update_jit_logic(
            self.freq_grid,
            self.linewidth_grid,
            self.posterior_2d,
            float(mx),
            float(my),
            float(uncert),
            float(amplitude),
            float(background),
            int(noise_model_code),
        )

    def calculate_utility(self, n_samples, amplitude, background, sigma):
        return _calculate_utility_jit_logic(
            self.freq_grid,
            self.linewidth_grid,
            self.posterior_2d,
            int(n_samples),
            float(amplitude),
            float(background),
            float(sigma),
        )

    def get_estimates(self):
        return _get_estimates_jit_logic(self.freq_grid, self.linewidth_grid, self.posterior_2d)
