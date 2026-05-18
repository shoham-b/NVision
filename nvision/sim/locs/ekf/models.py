import numpy as np
from numba import njit


@njit
def mu(f: float, theta: np.ndarray) -> float:
    """Triple-Lorentzian ODMR photoluminescence function.

    Args:
        f: Microwave frequency in MHz.
        theta: Parameter vector [f_B, delta_f_HF, a, Omega, k_NP].

    Returns:
        Photoluminescence response mu.
    """
    f_B = theta[0]  # noqa: N806
    delta_f_HF = theta[1]  # noqa: N806
    a = theta[2]
    Omega = theta[3]  # noqa: N806
    k_NP = theta[4]  # noqa: N806

    Omega2 = Omega**2  # noqa: N806
    D_minus = (f - f_B - delta_f_HF) ** 2 + Omega2  # noqa: N806
    D_zero = (f - f_B) ** 2 + Omega2  # noqa: N806
    D_plus = (f - f_B + delta_f_HF) ** 2 + Omega2  # noqa: N806

    return 1.0 - a * (k_NP / D_minus + 1.0 / D_zero + 1.0 / (k_NP * D_plus))


@njit
def jacobian_mu(f: float, theta: np.ndarray) -> np.ndarray:
    """Analytical Jacobian row vector (gradient of mu w.r.t theta).

    Args:
        f: Microwave frequency in MHz.
        theta: Parameter vector [f_B, delta_f_HF, a, Omega, k_NP].

    Returns:
        Jacobian row vector H (shape 1x5).
    """
    f_B = theta[0]  # noqa: N806
    delta_f_HF = theta[1]  # noqa: N806
    a = theta[2]
    Omega = theta[3]  # noqa: N806
    k_NP = theta[4]  # noqa: N806

    Omega2 = Omega**2  # noqa: N806
    D_minus = (f - f_B - delta_f_HF) ** 2 + Omega2  # noqa: N806
    D_zero = (f - f_B) ** 2 + Omega2  # noqa: N806
    D_plus = (f - f_B + delta_f_HF) ** 2 + Omega2  # noqa: N806

    D_minus2 = D_minus**2  # noqa: N806
    D_zero2 = D_zero**2  # noqa: N806
    D_plus2 = D_plus**2  # noqa: N806

    # Pre-compute components
    f_minus = f - f_B - delta_f_HF
    f_zero = f - f_B
    f_plus = f - f_B + delta_f_HF

    # Derivatives
    d_mu_d_f_B = -2.0 * a * (k_NP * f_minus / D_minus2 + f_zero / D_zero2 + f_plus / (k_NP * D_plus2))  # noqa: N806
    d_mu_d_delta_f_HF = -2.0 * a * (k_NP * f_minus / D_minus2 - f_plus / (k_NP * D_plus2))  # noqa: N806
    d_mu_d_a = -(k_NP / D_minus + 1.0 / D_zero + 1.0 / (k_NP * D_plus))
    d_mu_d_Omega = 2.0 * a * Omega * (k_NP / D_minus2 + 1.0 / D_zero2 + 1.0 / (k_NP * D_plus2))  # noqa: N806
    d_mu_d_k_NP = -a * (1.0 / D_minus - 1.0 / (k_NP**2 * D_plus))  # noqa: N806

    # Return as 1x5 2D array for proper matrix multiplication
    H = np.empty((1, 5), dtype=np.float64)  # noqa: N806
    H[0, 0] = d_mu_d_f_B
    H[0, 1] = d_mu_d_delta_f_HF
    H[0, 2] = d_mu_d_a
    H[0, 3] = d_mu_d_Omega
    H[0, 4] = d_mu_d_k_NP
    return H
