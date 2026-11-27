"""Run-time posterior metrics of the primary parameter (``zeeman_shift``), computed from the belief snapshots.

Nothing here touches the SMC, SBED or the likelihood: it only reads each snapshot's particle cloud. The cache keeps no
belief snapshots, so these are computed while a run is recorded (``runner/metrics.py``, ``runner/plots.py``) and stored
with it; a run recorded before this module existed has none of them and needs a re-run (``nv render`` cannot fill them).

Per step ``t`` (snapshot ``t`` = the belief after measurement ``t``), all in the physical units of the parameter (Hz):

* ``z_score``: signed standardised error ``(estimate - truth) / sigma_hat`` with ``sigma_hat`` the claimed uncertainty
  (``belief.uncertainty()``, the same value the run axes use). Calibrated and Gaussian: ``z_score ~ N(0, 1)``.
* ``pit``: the posterior CDF at the truth, the total weight of the particles with ``theta_i <= truth`` (in
  ``[0, 1]``). Calibrated: ``pit ~ U(0, 1)`` whatever the posterior's shape, so the share of runs with ``pit`` in
  the central 68 % / 95 % interval (``0.16 <= pit <= 0.84``, ``0.025 <= pit <= 0.975``) is the quantile coverage that
  needs no Gaussian peak (``claims.coverage_from_pit``). ``pit_final`` is ``pit`` at the last step.
* ``negentropy`` (nats): ``1/2 ln(2 pi e V) - H`` of the smoothed posterior density, ``V`` its own variance and ``H``
  its differential entropy. ``0`` for a Gaussian, ``> 0`` otherwise (a Gaussian maximises entropy at fixed variance).
  The brief's reference ``1/2 ln(2 pi e sigma_hat^2)`` is *not* used: the KDE adds the bandwidth to the variance, so
  against ``sigma_hat`` a Gaussian cloud would come out negative.
* ``far_mass``: posterior mass farther than ``FAR_FWHM`` (3) true FWHM from the posterior mode (mode of a
  weighted histogram with bins of FWHM/4).
* ``posterior_fisher``: Fisher information ``int p'(theta)^2 / p(theta) dtheta`` of the posterior density, Hz^-2;
  ``kde_bandwidth``: the KDE bandwidth it was computed with. ``negentropy``, ``posterior_fisher`` and ``kde_bandwidth``
  exist only at the steps the stored series keeps and at the detection step.
* ``far_placed``: 1 if measurement ``t`` was placed farther than ``FAR_FWHM`` FWHM from the mode of the belief *before*
  it
  (snapshot ``t - 1``), 0 if not, ``None`` for ``t = 0`` (no belief yet) and when the parameter is not
  ``zeeman_shift`` (the only one whose drive-axis position is known: the visible dip sits at ``zero_field_freq +
  zeeman_shift``). The mode is mapped onto the drive axis and compared with the drive frequency measured.

``j_tau_phys`` = ``posterior_fisher`` at the run's own detection step ``detection_step`` (snapshot ``tau - 1``): the
information of the posterior van Trees is restarted from (``articles/sections/sm_prior.tex``).

Density estimate (:func:`posterior_density`). A weighted Gaussian KDE evaluated on a grid spanning the particle support
(``min - 5 h_i`` to ``max + 5 h_i``, clipped to the parameter's bounds), by weighted binning then FFT convolution; the
derivative ``p'`` uses the derivative-of-Gaussian kernel. Edges: the density is *reflected* at the parameter bounds
(the SMC reflects particles that leave them), by mirror padding; at an edge the support does not reach, ``p`` is ~0 and
reflection is immaterial.

* Pilot bandwidth ``h``: the Botev-Grotowski-Kroese improved Sheather-Jones rule (``isj_bandwidth``) with the Kish
  effective sample size, *not* Silverman: Silverman's ``h ~ 1.06 sigma N^-1/5`` takes ``sigma`` from the whole cloud, so
  for the two mirror peaks right after detection (peaks far narrower than their separation) it smooths each peak to the
  width of the cloud and returns ``J ~ 1/h^2``, a bandwidth artefact. ISJ solves a fixed-point equation on the data and
  resolves the peaks.
* Final density: sample-point adaptive (Abramson), ``h_i = h (pilot(x_i) / g)^(-1/2)`` clamped to ``[1, 100] h``,
  because one global ``h`` tuned to narrow peaks turns a diffuse tail into spikes whose ``p'^2 / p`` inflates ``J``.

J depends on the smoothing (J falls roughly as ``1 / h^2`` where h is the limiting scale), so ``J_tau`` is good to about
a factor 2: ``kde_bandwidth_tau_phys`` is the pilot ``h`` and ``j_tau_2h_phys`` the same J with
twice that bandwidth, and the ``h`` series shows where the bandwidth moves (``docs/evaluation_metrics.md``).

Cost: a density takes ~25 ms, so ``g``, ``j``, ``h`` are computed only at the steps the stored series keeps (at most
``MAX_SERIES_POINTS``) and at tau; ``z``, ``m``, ``x`` need no density and cover every step.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.fft import dct
from scipy.optimize import brentq
from scipy.signal import fftconvolve

log = logging.getLogger(__name__)

FAR_FWHM = 3.0  # "far" = more than this many true FWHM from the posterior mode
_ISJ_BINS = 2**11  # bins of the ISJ bandwidth search (grid over the particle range)
_GRID_PER_BANDWIDTH = 5.0  # final grid spacing = h / this
_KERNEL_HALF_WIDTHS = 5.0  # kernel and grid padding, in bandwidths
_MAX_GRID_POINTS = 2**18
_LAMBDA_MIN, _LAMBDA_MAX = 1.0, 100.0  # clamp of the Abramson local-bandwidth factor
_TAIL_FLOOR = 1e-10  # p below this fraction of its max is dropped from p'^2/p (underflow noise, mass < 1e-9)

# Per-run scalar keys (results-row columns); see :func:`compute_posterior_metrics`.
POSTERIOR_METRIC_KEYS: tuple[str, ...] = (
    "j_tau_phys",
    "kde_bandwidth_tau_phys",
    "negentropy_tau",
    "far_mass_tau",
    "far_placed_steps",
    "far_placed_share",
    "j_tau_2h_phys",
    "undefined_posterior_steps",
    "pit_final",
)
# Per-step series keys (scan-entry ``series``), aligned with ``"step"``.
POSTERIOR_SERIES_KEYS: tuple[str, ...] = (
    "z_score",
    "pit",
    "negentropy",
    "far_mass",
    "posterior_fisher",
    "kde_bandwidth",
    "far_placed",
)


class NoBandwidthError(ValueError):
    """The ISJ equation has no solution: the cloud carries no structure to smooth (e.g. a flat, un-updated prior)."""


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
        except (ValueError, ZeroDivisionError):  # no sign change / vanishing functional: no root
            continue
    if t is None or not t > 0.0:
        raise NoBandwidthError("The ISJ bandwidth equation has no root on this sample (degenerate particle cloud?).")
    return math.sqrt(t) * r


def posterior_density(
    x_phys: np.ndarray, weights: np.ndarray, lo_phys: float, hi_phys: float, bandwidth_scale: float = 1.0
) -> PosteriorDensity:
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
    h = bandwidth_scale * isj_bandwidth(x, w, max(lo_phys, x_min - pad), min(hi_phys, x_max + pad))

    # Pilot: fixed bandwidth h (resolves the narrow structure). Final: sample-point adaptive (Abramson), where each
    # particle's bandwidth is h * (pilot(x_i) / g)^(-1/2), g the weighted geometric mean of the pilot. A single global
    # h tuned to narrow peaks leaves a diffuse tail (a few sparse particles) as spikes, whose p'^2 / p inflates J.
    grid, pilot, _ = _kde(x, w, np.full(x.shape, h), lo_phys, hi_phys)
    pilot_at_x = np.maximum(np.interp(x, grid, pilot), 1e-300)
    geo_mean = math.exp(float(np.sum(w * np.log(pilot_at_x))))
    lam = np.clip((pilot_at_x / geo_mean) ** -0.5, _LAMBDA_MIN, _LAMBDA_MAX)
    # Quantised to half-octave steps so the kernels fall into a few classes (one convolution per class).
    lam = 2.0 ** (np.round(2.0 * np.log2(lam)) / 2.0)
    grid, pdf, dpdf = _kde(x, w, h * lam, lo_phys, hi_phys)
    return PosteriorDensity(grid, pdf, dpdf, h)


def _kde(
    x: np.ndarray, w: np.ndarray, h_i: np.ndarray, lo_phys: float, hi_phys: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-point-adaptive Gaussian KDE and its derivative on a grid, reflected at the bounds.

    ``x``, ``w``, ``h_i``: shape (n_particles,), positions (Hz), normalised weights and per-particle bandwidths (Hz).
    Returns ``(grid_phys, pdf, dpdf)``, each shape (n_grid,): grid (Hz), density (1/Hz), derivative (1/Hz^2). Particles
    are grouped by bandwidth (``h_i`` takes few distinct values) and each group is binned and convolved with its own
    kernel, so the cost is one FFT convolution per distinct bandwidth, not per particle.
    """
    lo = max(lo_phys, float(np.min(x - _KERNEL_HALF_WIDTHS * h_i)))
    hi = min(hi_phys, float(np.max(x + _KERNEL_HALF_WIDTHS * h_i)))
    h_min = float(h_i.min())
    n_grid = math.ceil((hi - lo) / (h_min / _GRID_PER_BANDWIDTH))
    if n_grid > _MAX_GRID_POINTS:
        raise ValueError(f"The KDE grid would need {n_grid} points (h = {h_min} Hz over {hi - lo} Hz).")
    n_grid = max(n_grid, 16)
    dx = (hi - lo) / n_grid
    edges = np.linspace(lo, hi, n_grid + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    pdf = np.zeros(n_grid)
    dpdf = np.zeros(n_grid)
    for h in np.unique(h_i):
        sel = h_i == h
        hist, _ = np.histogram(x[sel], bins=edges, weights=w[sel])
        half = math.ceil(_KERNEL_HALF_WIDTHS * h / dx)
        offs = np.arange(-half, half + 1) * dx
        gauss = np.exp(-0.5 * (offs / h) ** 2) / (h * math.sqrt(2.0 * math.pi))
        dgauss = -offs / h**2 * gauss
        # Mirror padding = reflection at the grid edges (half-sample symmetric: bins are cell-centred). ``hist``
        # holds probability per bin and the kernels are densities in 1/Hz, so the convolution is already a density.
        padded = np.pad(hist, half, mode="symmetric")
        pdf += fftconvolve(padded, gauss, mode="valid")
        dpdf += fftconvolve(padded, dgauss[::-1], mode="valid")
    return grid, pdf, dpdf


def _to_physical_particles(belief: Any, primary: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | None:
    """``(positions_phys, weights, bounds_phys)`` of ``primary``, or ``None`` for a belief with no particle cloud."""
    weights_fn = getattr(belief, "weights", None)
    particles_fn = getattr(belief, "particles_phys", None)
    if not callable(weights_fn) or not callable(particles_fn):
        return None
    return particles_fn()[primary], weights_fn(), belief.physical_param_bounds[primary]


def _coarse_mode(x: np.ndarray, w: np.ndarray, bin_width: float) -> float:
    """Mode (Hz) of the weighted histogram of ``x`` with bins of ``bin_width`` Hz (shape: (n_particles,))."""
    n_bins = max(1, math.ceil((float(x.max()) - float(x.min())) / bin_width))
    hist, edges = np.histogram(x, bins=n_bins, weights=w)
    i = int(np.argmax(hist))
    return 0.5 * float(edges[i] + edges[i + 1])


class _StepPosterior:
    """One distinct belief's summary of the primary parameter; the density fields are computed on demand.

    ``informed`` is False for a belief no measurement has updated yet (a flat prior, as a sweep's belief is until its
    first flush): it has no posterior, so every field is ``None`` and the step is counted in
    ``undefined_posterior_steps``. ``mode_phys`` and ``far_mass`` are cheap (a weighted histogram with bins of a quarter
    FWHM, so the mode is within FWHM/8 of the KDE's, far inside the 3 FWHM threshold) and exist for every informed
    step. The KDE fields (``fisher``, ``negentropy``, ``bandwidth_phys``) cost ~25 ms and only exist after
    :meth:`ensure_density`.
    """

    def __init__(self, belief: Any, primary: str, fwhm_phys: float | None) -> None:
        got = _to_physical_particles(belief, primary)
        if got is None:
            raise ValueError("Belief has no particle cloud.")
        self._x, w, self._bounds = got
        self._w = w / w.sum()
        self.informed = belief.num_observations > 0
        self.mode_phys = self.far_mass = self.fisher = self.negentropy = self.bandwidth_phys = None
        self._has_density = False
        self.no_bandwidth = False
        if self.informed and fwhm_phys is not None:
            self.mode_phys = _coarse_mode(self._x, self._w, fwhm_phys / 4.0)
            self.far_mass = float(np.sum(self._w[np.abs(self._x - self.mode_phys) > FAR_FWHM * fwhm_phys]))

    def pit(self, truth_phys: float) -> float:
        """Posterior CDF at the truth: total weight (shape: (n_particles,)) of particles <= ``truth_phys`` (Hz)."""
        return float(np.sum(self._w[self._x <= truth_phys]))

    def ensure_density(self) -> None:
        if self._has_density or self.no_bandwidth or not self.informed:
            return
        try:
            density = posterior_density(self._x, self._w, *self._bounds)
        except NoBandwidthError:
            # A cloud with no structure to smooth (e.g. still the flat prior after the first measurements): the
            # density-based fields are undefined there. Recorded as missing, counted in undefined_posterior_steps.
            self.no_bandwidth = True
            return
        self.bandwidth_phys = density.bandwidth_phys
        self.fisher = density.fisher_information()
        self.negentropy = density.negentropy()
        self._has_density = True

    def fisher_at_bandwidth_scale(self, scale: float) -> float:
        """J (Hz^-2) with the bandwidth multiplied by ``scale``: how much J is set by the smoothing."""
        return posterior_density(self._x, self._w, *self._bounds, bandwidth_scale=scale).fisher_information()


def compute_posterior_metrics(
    run_result: Any,
    primary: str | None,
    axes: dict[str, Any],
    max_points: int | None = None,
) -> dict[str, Any]:
    """Posterior metrics of ``primary`` over one run: ``{"scalars", "series", "density_steps", "max_points"}``.

    ``axes``: the dict :func:`~nvision.metrics.run_axes.compute_run_axes` returned for the run
    (``detection_step``). ``scalars`` has every
    :data:`POSTERIOR_METRIC_KEYS` key; ``series`` every :data:`POSTERIOR_SERIES_KEYS` key as a list of length
    ``n_steps`` (``None`` where undefined). Everything is ``None`` when there is no ``primary`` or the beliefs carry no
    particle cloud (grid beliefs). Memoised on ``run_result.posterior_cache``.

    ``z``, ``p``, ``m`` and ``x`` exist at every step (cheap). The KDE series ``g`` and ``j`` are computed only at the
    steps the stored series keeps (``density_steps``, 0-based: the ``max_points`` indices
    :func:`~nvision.metrics.series._downsample_indices` picks, default ``MAX_SERIES_POINTS``) and at the detection
    step, because a density costs ~25 ms and a budget-length run would otherwise spend over a minute on them.
    """
    from nvision.metrics.series import MAX_SERIES_POINTS, _downsample_indices

    max_points = MAX_SERIES_POINTS if max_points is None else max_points
    cached = getattr(run_result, "posterior_cache", None)
    if cached is not None:
        if cached["max_points"] != max_points:
            raise ValueError(f"Posterior metrics were built for max_points={cached['max_points']}, not {max_points}.")
        return cached
    snapshots = run_result.snapshots
    n = len(snapshots)
    scalars: dict[str, Any] = {k: None for k in POSTERIOR_METRIC_KEYS}
    series: dict[str, list[float | None]] = {k: [None] * n for k in POSTERIOR_SERIES_KEYS}
    out: dict[str, Any] = {
        "scalars": scalars,
        "series": series,
        "density_steps": [],
        "max_points": max_points,
        "computed": False,  # True once the per-step series were built (not for no primary / no particle cloud)
    }
    if n == 0 or primary is None or _to_physical_particles(snapshots[0].belief, primary) is None:
        return out

    from nvision.metrics.run_axes import _true_fwhm_phys  # the same FWHM the run axes use

    true_signal = run_result.true_signal
    truth = run_result.truth_trace(primary)  # shape: (n_steps,), Hz
    fwhm = _true_fwhm_phys(true_signal)  # None for drive-controlled models (width depends on the drive)
    zff_true = float(true_signal.get_param_value("zero_field_freq")) if primary == "zeeman_shift" else None

    # The observer shares one belief copy across steps a locator buffers (sweeps), so each distinct belief is
    # summarised once.
    by_belief: dict[int, _StepPosterior] = {}
    posts = []
    for s in snapshots:
        if id(s.belief) not in by_belief:
            by_belief[id(s.belief)] = _StepPosterior(s.belief, primary, fwhm)
        posts.append(by_belief[id(s.belief)])

    tau = axes.get("detection_step")
    density_steps = set(_downsample_indices(n, max_points))
    if tau is not None:
        density_steps.add(int(tau) - 1)
    out["density_steps"] = sorted(density_steps)
    out["computed"] = True

    for t, (snap, post) in enumerate(zip(snapshots, posts, strict=True)):
        est = float(snap.belief.estimates()[primary])
        sigma = float(snap.belief.uncertainty()[primary])
        if not sigma > 0.0:
            raise ValueError(f"Claimed uncertainty of {primary!r} is {sigma} at step {t + 1}; cannot standardise.")
        series["z_score"][t] = (est - truth[t]) / sigma
        series["pit"][t] = post.pit(truth[t])
        series["far_mass"][t] = post.far_mass
        if t in density_steps:
            post.ensure_density()
            series["negentropy"][t] = post.negentropy
            series["posterior_fisher"][t] = post.fisher
            series["kde_bandwidth"][t] = post.bandwidth_phys
        if t >= 1 and primary == "zeeman_shift" and fwhm is not None and posts[t - 1].informed:
            prev = snapshots[t - 1].belief
            lo, hi = prev.drive_freq_bounds_phys
            drive_phys = lo + snap.obs.drive_freq_unit * (hi - lo)
            # Visible dip at zero_field_freq + zeeman_shift: fixed (the true value) unless the belief infers it.
            zff = float(prev.estimates().get("zero_field_freq", zff_true))
            series["far_placed"][t] = float(abs(drive_phys - (zff + posts[t - 1].mode_phys)) > FAR_FWHM * fwhm)

    # Steps with no posterior density: no measurement had updated the belief, or its cloud has no structure for
    # the bandwidth rule (checked at the steps a density is computed for).
    scalars["pit_final"] = series["pit"][-1]
    scalars["undefined_posterior_steps"] = sum((not p.informed) or p.no_bandwidth for p in posts)
    if tau is not None:
        post = posts[int(tau) - 1]  # the density fields are None when no measurement had updated the belief yet
        scalars["j_tau_phys"] = post.fisher
        scalars["kde_bandwidth_tau_phys"] = post.bandwidth_phys
        if post.fisher is not None:
            scalars["j_tau_2h_phys"] = post.fisher_at_bandwidth_scale(2.0)
        scalars["negentropy_tau"] = post.negentropy
        scalars["far_mass_tau"] = post.far_mass
    placed = [(t, v) for t, v in enumerate(series["far_placed"]) if v is not None]
    if placed:
        scalars["far_placed_steps"] = int(sum(v for _, v in placed))
        scalars["far_placed_share"] = float(np.mean([v for _, v in placed]))
    run_result.posterior_cache = out
    return out
