"""Run-time posterior metrics of the primary parameter (``zeeman_shift``), computed from the belief snapshots.

Nothing here touches the SMC, SBED or the likelihood: it only reads each snapshot's particle cloud. The cache keeps no
belief snapshots, so these are computed while a run is recorded (``runner/metrics.py``, ``runner/plots.py``) and stored
with it; a run recorded before this module existed has none of them and needs a re-run (``nv render`` cannot fill them).

Per step ``t`` (snapshot ``t`` = the belief after measurement ``t``), all in the physical units of the parameter (Hz):

* ``z``: signed standardised error ``(estimate - truth) / sigma_hat`` with ``sigma_hat`` the claimed uncertainty
  (``belief.uncertainty()``, the same value the stage metrics use). Calibrated and Gaussian: ``z ~ N(0, 1)``.
* ``g`` (negentropy, nats): ``1/2 ln(2 pi e V) - H`` of the smoothed posterior density, ``V`` its own variance and ``H``
  its differential entropy. ``0`` for a Gaussian, ``> 0`` otherwise (a Gaussian maximises entropy at fixed variance).
  The brief's reference ``1/2 ln(2 pi e sigma_hat^2)`` is *not* used: the KDE adds the bandwidth to the variance, so
  against ``sigma_hat`` a Gaussian cloud would come out negative.
* ``m`` (far mass): posterior mass farther than ``FAR_FWHM`` (3) true FWHM from the posterior mode (KDE argmax).
* ``j``: Fisher information ``int p'(theta)^2 / p(theta) dtheta`` of the posterior density, Hz^-2.
* ``x``: 1 if measurement ``t`` was placed farther than ``FAR_FWHM`` FWHM from the mode of the belief *before* it
  (snapshot ``t - 1``), 0 if not, ``None`` for ``t = 0`` (no belief yet) and when the parameter is not
  ``zeeman_shift`` (the only one whose drive-axis position is known: the visible dip sits at ``zero_field_freq +
  zeeman_shift``). The mode is mapped onto the drive axis and compared with the drive frequency measured.

``J_tau`` = ``j`` at the run's own detection step ``dip_detected_step`` (snapshot ``tau - 1``): the information of the
posterior van Trees is restarted from (``articles/sections/sm_prior.tex``).

Density estimate (:func:`posterior_density`). A weighted Gaussian KDE evaluated on a grid spanning the particle support
(``min - 5 h`` to ``max + 5 h``, clipped to the parameter's bounds), by weighted binning then Gaussian convolution; the
derivative ``p'`` uses the derivative-of-Gaussian kernel. Edges: the density is *reflected* at the parameter bounds
(the SMC reflects particles that leave them), by mirror padding; at an edge the support does not reach, ``p`` is ~0 and
reflection is immaterial. Bandwidth ``h`` by the Botev-Grotowski-Kroese improved Sheather-Jones rule
(``isj_bandwidth``), with the Kish effective sample size, *not* Silverman: Silverman's ``h ~ 1.06 sigma N^-1/5`` takes
``sigma`` from the whole cloud, so for the two mirror peaks right after detection (peaks far narrower than their
separation) it smooths each peak to the width of the cloud and returns ``J ~ 1/h^2``, a bandwidth artefact. ISJ solves
a fixed-point equation on the data and resolves the peaks. The bandwidth is stored next to ``J_tau``
(``posterior_kde_bandwidth_tau_phys``) so a
bandwidth-limited value is visible.

Cost: one posterior density per step (a few ms, see ``docs/evaluation_metrics.md``), computed on every step.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.fft import dct
from scipy.optimize import brentq

log = logging.getLogger(__name__)

FAR_FWHM = 3.0  # "far" = more than this many true FWHM from the posterior mode
_ISJ_BINS = 2**11  # bins of the ISJ bandwidth search (grid over the particle range)
_GRID_PER_BANDWIDTH = 5.0  # final grid spacing = h / this
_KERNEL_HALF_WIDTHS = 5.0  # kernel and grid padding, in bandwidths
_MAX_GRID_POINTS = 2**18
_TAIL_FLOOR = 1e-10  # p below this fraction of its max is dropped from p'^2/p (underflow noise, mass < 1e-9)

# Per-run scalar keys (results-row columns); see :func:`compute_posterior_metrics`.
POSTERIOR_METRIC_KEYS: tuple[str, ...] = (
    "posterior_J_tau_phys",
    "posterior_kde_bandwidth_tau_phys",
    "posterior_negentropy_tau",
    "posterior_far_mass_tau",
    "posterior_far_placed_steps",
    "posterior_far_placed_share",
    "posterior_far_placed_share_stage2",
)
# Per-step series keys (scan-entry ``series``), aligned with ``"s"``.
POSTERIOR_SERIES_KEYS: tuple[str, ...] = ("z", "g", "m", "j", "x")


class PosteriorDensity:
    """Smoothed posterior density of one parameter on a grid.

    ``grid_phys``, ``pdf``, ``dpdf``: shape (n_grid,), grid (Hz), density (1/Hz) and its derivative (1/Hz^2).
    ``bandwidth_phys``: the KDE bandwidth h (Hz).
    """

    def __init__(self, grid_phys: np.ndarray, pdf: np.ndarray, dpdf: np.ndarray, bandwidth_phys: float) -> None:
        self.grid_phys = grid_phys
        self.pdf = pdf
        self.dpdf = dpdf
        self.bandwidth_phys = bandwidth_phys
        self.dx = float(grid_phys[1] - grid_phys[0])

    def fisher_information(self) -> float:
        """``int p'^2 / p`` (Hz^-2) of the density."""
        keep = self.pdf > _TAIL_FLOOR * float(self.pdf.max())
        return float(np.sum(self.dpdf[keep] ** 2 / self.pdf[keep]) * self.dx)

    def entropy(self) -> float:
        """Differential entropy ``-int p ln p`` (nats, with x in Hz)."""
        p = self.pdf[self.pdf > 0.0]
        return float(-np.sum(p * np.log(p)) * self.dx)

    def variance(self) -> float:
        """Variance (Hz^2) of the density, the bandwidth included."""
        mean = float(np.sum(self.grid_phys * self.pdf) * self.dx)
        return float(np.sum((self.grid_phys - mean) ** 2 * self.pdf) * self.dx)

    def negentropy(self) -> float:
        """``1/2 ln(2 pi e V) - H`` (nats): 0 for a Gaussian, > 0 otherwise."""
        return 0.5 * math.log(2.0 * math.pi * math.e * self.variance()) - self.entropy()

    def mode(self) -> float:
        """Grid argmax of the density (Hz)."""
        return float(self.grid_phys[int(np.argmax(self.pdf))])


def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    mean = float(np.sum(w * x))
    return math.sqrt(float(np.sum(w * (x - mean) ** 2)))


def isj_bandwidth(x: np.ndarray, w: np.ndarray, lo: float, hi: float) -> float:
    """Improved Sheather-Jones bandwidth (Hz) of the weighted sample ``x`` (Botev et al. 2010).

    ``x``, ``w``: shape (n_particles,), positions (Hz) and normalised weights. ``lo``, ``hi``: the span the sample is
    binned over (Hz). Solves the ISJ fixed-point equation on the DCT of the binned density, with the Kish effective
    sample size ``1 / sum w^2`` as ``N``. Raises ``ValueError`` when the equation has no root (e.g. a degenerate cloud).
    """
    n_eff = 1.0 / float(np.sum(w**2))
    r = hi - lo
    hist, _ = np.histogram(x, bins=_ISJ_BINS, range=(lo, hi), weights=w)
    a = dct(hist / hist.sum(), type=2)
    a[0] *= 0.5  # (Botev's unnormalised cosine transform: a_0 = sum of the data = 1)
    k2 = np.arange(1, _ISJ_BINS, dtype=np.float64) ** 2
    a2 = (a[1:] / 2.0) ** 2
    pi2 = math.pi**2
    ell = 7
    terms = {s: k2**s * a2 for s in range(2, ell + 1)}  # k^(2s) a_k^2, reused by every root-finder evaluation
    consts = {
        s: (
            float(np.prod(np.arange(1, 2 * s, 2))) / math.sqrt(2.0 * math.pi),  # K0
            (1.0 + 0.5 ** (s + 0.5)) / 3.0,
        )
        for s in range(2, ell)
    }

    def functional(s: int, time: float) -> float:
        """``2 pi^(2s) sum_k k^(2s) a_k^2 exp(-k^2 pi^2 time)``; terms past exp(-40) are dropped."""
        kmax = min(len(k2), int(math.sqrt(40.0 / (pi2 * time))) + 1)
        return 2.0 * math.pi ** (2 * s) * float(np.dot(terms[s][:kmax], np.exp(-k2[:kmax] * (pi2 * time))))

    def fixed_point(t: float) -> float:
        f = functional(ell, t)
        for s in range(ell - 1, 1, -1):
            k0, const = consts[s]
            f = functional(s, (2.0 * const * k0 / n_eff / f) ** (2.0 / (3.0 + 2.0 * s)))
        return t - (2.0 * n_eff * math.sqrt(math.pi) * f) ** (-2.0 / 5.0)

    t = None
    for t_hi in (0.1, 0.5, 1.0):
        try:
            t = brentq(fixed_point, 1e-12, t_hi, xtol=1e-30, rtol=1e-3)
            break
        except ValueError:
            continue
    if t is None or not t > 0.0:
        raise ValueError("The ISJ bandwidth equation has no root on this sample (degenerate particle cloud?).")
    return math.sqrt(t) * r


def posterior_density(x_phys: np.ndarray, weights: np.ndarray, lo_phys: float, hi_phys: float) -> PosteriorDensity:
    """Weighted Gaussian KDE of a particle posterior with reflection at ``[lo_phys, hi_phys]``.

    ``x_phys``, ``weights``: shape (n_particles,), particle positions (Hz) inside the bounds and their weights (any
    positive scale). See the module docstring for the bandwidth rule and the grid. Raises ``ValueError`` for a
    particle outside the bounds, non-positive weights or a cloud with no spread.
    """
    x = np.asarray(x_phys, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.shape != w.shape or x.ndim != 1:
        raise ValueError(f"x and weights must be 1-D of equal length, got {x.shape} and {w.shape}.")
    if not np.all(w >= 0.0) or not float(w.sum()) > 0.0:
        raise ValueError("Particle weights must be non-negative with a positive sum.")
    w = w / w.sum()
    if x.min() < lo_phys - 1e-9 * (hi_phys - lo_phys) or x.max() > hi_phys + 1e-9 * (hi_phys - lo_phys):
        raise ValueError(f"Particles span [{x.min()}, {x.max()}], outside the bounds [{lo_phys}, {hi_phys}].")
    if not _weighted_std(x, w) > 0.0:
        raise ValueError("The particle cloud has no spread, so it has no density.")

    x_min, x_max = float(x.min()), float(x.max())
    pad = 0.05 * (x_max - x_min)
    h = isj_bandwidth(x, w, max(lo_phys, x_min - pad), min(hi_phys, x_max + pad))

    lo = max(lo_phys, x_min - _KERNEL_HALF_WIDTHS * h)
    hi = min(hi_phys, x_max + _KERNEL_HALF_WIDTHS * h)
    n_grid = math.ceil((hi - lo) / (h / _GRID_PER_BANDWIDTH))
    if n_grid > _MAX_GRID_POINTS:
        raise ValueError(f"The KDE grid would need {n_grid} points (h = {h} Hz over {hi - lo} Hz).")
    n_grid = max(n_grid, 16)
    dx = (hi - lo) / n_grid
    hist, edges = np.histogram(x, bins=n_grid, range=(lo, hi), weights=w)
    grid = 0.5 * (edges[:-1] + edges[1:])

    half = math.ceil(_KERNEL_HALF_WIDTHS * h / dx)
    offs = np.arange(-half, half + 1) * dx
    gauss = np.exp(-0.5 * (offs / h) ** 2) / (h * math.sqrt(2.0 * math.pi))
    dgauss = -offs / h**2 * gauss
    # Mirror padding = reflection at the edges of the grid (half-sample symmetric: bins are cell-centred).
    padded = np.pad(hist, half, mode="symmetric")
    # ``hist`` holds probability per bin and the kernels are densities in 1/Hz, so convolve(hist, gauss) is already
    # the density (sum over bins of mass * kernel); no division by dx.
    pdf = np.convolve(padded, gauss, mode="valid")
    dpdf = np.convolve(padded, dgauss[::-1], mode="valid")
    return PosteriorDensity(grid, pdf, dpdf, h)


def _to_physical_particles(belief: Any, primary: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | None:
    """``(positions_phys, weights, bounds_phys)`` of ``primary``, or ``None`` for a belief with no particle cloud."""
    weights_fn = getattr(belief, "weights", None)
    particles_fn = getattr(belief, "particles_phys", None)
    if not callable(weights_fn) or not callable(particles_fn):
        return None
    return particles_fn()[primary], weights_fn(), belief.physical_param_bounds[primary]


class _StepPosterior:
    """One snapshot's posterior of the primary parameter: the KDE, its mode and the far mass."""

    def __init__(self, belief: Any, primary: str, fwhm_phys: float | None) -> None:
        got = _to_physical_particles(belief, primary)
        if got is None:
            raise ValueError("Belief has no particle cloud.")
        x, w, (lo, hi) = got
        w = w / w.sum()
        self.density = posterior_density(x, w, lo, hi)
        self.mode_phys = self.density.mode()
        self.far_mass = (
            None if fwhm_phys is None else float(np.sum(w[np.abs(x - self.mode_phys) > FAR_FWHM * fwhm_phys]))
        )


def compute_posterior_metrics(run_result: Any, primary: str | None, stage_metrics: dict[str, Any]) -> dict[str, Any]:
    """Posterior metrics of ``primary`` over one run: ``{"scalars": {...}, "series": {...}}``.

    ``stage_metrics``: the dict :func:`~nvision.metrics.stages.compute_stage_metrics` returned for the run
    (``dip_detected_step``, ``focus_found_step``, ``refine_start_step``). ``scalars`` has every
    :data:`POSTERIOR_METRIC_KEYS` key; ``series`` every :data:`POSTERIOR_SERIES_KEYS` key as a list of length
    ``n_steps`` (``None`` where undefined). Everything is ``None`` when there is no ``primary`` or the beliefs carry no
    particle cloud (grid beliefs). Memoised on ``run_result.posterior_cache``.
    """
    cached = getattr(run_result, "posterior_cache", None)
    if cached is not None:
        return cached
    snapshots = run_result.snapshots
    n = len(snapshots)
    scalars: dict[str, Any] = {k: None for k in POSTERIOR_METRIC_KEYS}
    series: dict[str, list[float | None]] = {k: [None] * n for k in POSTERIOR_SERIES_KEYS}
    out = {"scalars": scalars, "series": series}
    if n == 0 or primary is None or _to_physical_particles(snapshots[0].belief, primary) is None:
        return out

    from nvision.metrics.stages import _true_fwhm_phys  # the same FWHM the stage axes use

    true_signal = run_result.true_signal
    truth = float(true_signal.get_param_value(primary))
    fwhm = _true_fwhm_phys(true_signal)  # None for drive-controlled models (width depends on the drive)
    zff_true = float(true_signal.get_param_value("zero_field_freq")) if primary == "zeeman_shift" else None

    posts = [_StepPosterior(s.belief, primary, fwhm) for s in snapshots]
    for t, (snap, post) in enumerate(zip(snapshots, posts, strict=True)):
        est = float(snap.belief.estimates()[primary])
        sigma = float(snap.belief.uncertainty()[primary])
        if not sigma > 0.0:
            raise ValueError(f"Claimed uncertainty of {primary!r} is {sigma} at step {t + 1}; cannot standardise.")
        series["z"][t] = (est - truth) / sigma
        series["g"][t] = post.density.negentropy()
        series["m"][t] = post.far_mass
        series["j"][t] = post.density.fisher_information()
        if t >= 1 and primary == "zeeman_shift" and fwhm is not None:
            prev = snapshots[t - 1].belief
            lo, hi = prev.drive_freq_bounds_phys
            drive_phys = lo + snap.obs.drive_freq_unit * (hi - lo)
            # Visible dip at zero_field_freq + zeeman_shift: fixed (the true value) unless the belief infers it.
            zff = float(prev.estimates().get("zero_field_freq", zff_true))
            series["x"][t] = float(abs(drive_phys - (zff + posts[t - 1].mode_phys)) > FAR_FWHM * fwhm)

    tau = stage_metrics.get("dip_detected_step")
    if tau is not None:
        post = posts[int(tau) - 1]
        scalars["posterior_J_tau_phys"] = series["j"][int(tau) - 1]
        scalars["posterior_kde_bandwidth_tau_phys"] = post.density.bandwidth_phys
        scalars["posterior_negentropy_tau"] = series["g"][int(tau) - 1]
        scalars["posterior_far_mass_tau"] = post.far_mass
    placed = [(t, v) for t, v in enumerate(series["x"]) if v is not None]
    if placed:
        scalars["posterior_far_placed_steps"] = int(sum(v for _, v in placed))
        scalars["posterior_far_placed_share"] = float(np.mean([v for _, v in placed]))
        focus, refine = stage_metrics.get("focus_found_step"), stage_metrics.get("refine_start_step")
        if focus is not None:  # localize = measurements focus+1 .. refine-1 (1-indexed), i.e. 0-based focus .. refine-2
            last = (int(refine) - 2) if refine is not None else n - 1
            loc = [v for t, v in placed if int(focus) <= t <= last]
            scalars["posterior_far_placed_share_stage2"] = float(np.mean(loc)) if loc else None
    run_result.posterior_cache = out
    return out
