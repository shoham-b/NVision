"""One record per run, shared by the simulated and the lab analyses.

Simulation and lab are the same metrics fed by two sources. Everything a metric needs about one run is in a
:class:`RunRecord`; where the two sources must differ, the difference is an explicit, named field of the record (never
an ``if lab:`` branch inside a metric):

1. **The truth** (:class:`TruthSource`). Simulation: the true parameter, drawn from the prior, so quantities averaged
   over the prior ensemble (the MSE against the bounds, rho, the fitted rate, the PIT) are valid
   (``ensemble_valid``). Lab: the reference sweep fit, whose standard error adds in quadrature and whose drift between
   recordings (0.4-0.6 MHz) floors every error, so only claimed quantities are valid (the claimed rate, the
   placement, the claimed-sigma search cost and the scatter of repeated runs).
2. **The detection** (:class:`Detector`). Simulation: the run's dip test fires (``DIP_TEST``, stored at run time).
   Lab: the first steep, lasting drop of the claimed sigma (``SIGMA_DROP``). ``SIGMA_DROP`` needs only the claimed
   sigma, so it runs on simulated runs too and the two definitions are compared on the same data
   (:func:`compare_detectors`).
3. **Nuisances** (``known_nuisances``): the lab holds the contrast and the width fixed, the simulation infers them. A
   label and an input of ``I_max`` only.
4. **Stop and budget** (``stop_rule``, ``budget``): simulated runs measure the whole budget; lab sets either stop at a
   claimed-sigma target or measure their full budget.

Arrays: ``steps`` (m,) are the stored measurement counts (1-based, ascending; a simulated run keeps at most 80 of
them), ``estimate``, ``claimed_sigma`` and ``error`` are aligned with them, in the physical unit of the parameter
(Hz for the simulation, MHz for the lab, as each source stores it; every metric is a ratio or uses the same unit
throughout). ``drive_freq`` and ``outcomes`` (n,) are every measurement, when the source keeps them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from nvision.metrics.lab_search_fit import detection_step as sigma_drop_step


@dataclass(frozen=True)
class TruthSource:
    """What a run's error is measured against, and whether quantities averaged over the prior ensemble are valid."""

    name: str
    ensemble_valid: bool
    description: str


SIM_TRUTH = TruthSource(
    "sim-truth",
    ensemble_valid=True,
    description="the true parameter of the simulated signal, drawn uniformly over the prior window",
)
LAB_REFERENCE = TruthSource(
    "lab-reference",
    ensemble_valid=False,
    description="the reference fit of the lab's sweeps; its standard error adds in quadrature and the line drifts "
    "0.4-0.6 MHz between recordings",
)


@dataclass(frozen=True)
class RunRecord:
    """One run, as every metric of the article reads it (see the module docstring for the explicit differences)."""

    source: str  # "sim" or "lab"
    truth: TruthSource
    steps: np.ndarray  # (m,) stored measurement counts, 1-based, ascending
    estimate: np.ndarray | None  # (m,) posterior mean of the primary parameter
    claimed_sigma: np.ndarray  # (m,) claimed sigma of the primary parameter, > 0
    error: np.ndarray  # (m,) |estimate - truth| (sim) or |estimate - reference| (lab), >= 0
    i_max: float  # Fisher information of one measurement at the steepest slope, unit^-2
    window: float  # W, the drive_freq window
    fwhm: float  # Gamma, the line's FWHM
    snr: float | None  # C / sigma of the line (article convention)
    budget: int  # measurements the run may take
    stop_rule: str  # "full-budget" or "claimed-sigma target"
    known_nuisances: bool  # contrast and width held fixed (lab) rather than inferred (sim)
    dip_test_step: int | None = None  # 1-based step the run's own dip test fired (sim only, stored at run time)
    drive_freq: np.ndarray | None = None  # (n,) every measurement's drive_freq
    outcomes: np.ndarray | None = None  # (n,) every measurement's outcome
    negentropy: np.ndarray | None = None  # (m,) posterior negentropy, nats (sim only: the lab keeps no particles)
    j_tau: float | None = None  # Fisher information of the posterior at the dip test (sim only), unit^-2
    far_placed: np.ndarray | None = None  # (m,) 1 if the measurement was far from the belief's mode
    far_mass_tau: float | None = None
    line_params: dict = field(default_factory=dict)  # the line-shape parameters (truth or the lab's fixed model)

    def __post_init__(self) -> None:
        s, c, e = self.steps, self.claimed_sigma, self.error
        if s.ndim != 1 or s.size == 0 or c.shape != s.shape or e.shape != s.shape:
            raise ValueError(
                f"steps, claimed_sigma, error must be equal 1-D arrays, got {s.shape}, {c.shape}, {e.shape}"
            )
        if self.estimate is not None and self.estimate.shape != s.shape:
            raise ValueError(f"estimate must have the shape of steps {s.shape}, got {self.estimate.shape}")
        if not np.all(np.diff(s) > 0):
            raise ValueError("steps must be strictly ascending")
        if self.source not in ("sim", "lab"):
            raise ValueError(f"source must be 'sim' or 'lab', got {self.source!r}")
        if not (self.i_max > 0 and self.window > 0 and self.fwhm > 0):
            raise ValueError(f"i_max, window, fwhm must be positive, got {self.i_max}, {self.window}, {self.fwhm}")


@dataclass(frozen=True)
class Detector:
    """A detection definition: the 1-based step that ends search + localize for one run, or ``None``."""

    name: str
    description: str
    step: Callable[[RunRecord], int | None]


def _dip_test(record: RunRecord) -> int | None:
    if record.source != "sim":
        raise ValueError("the dip-test detection is stored only for simulated runs")
    return record.dip_test_step


def _sigma_drop(record: RunRecord) -> int | None:
    """``lab_search_fit.detection_step`` on the stored claimed sigma, mapped back to a measurement count.

    A simulated run stores a subsample of its steps, so its drop is located to the stored step after it.
    """
    k = sigma_drop_step(record.claimed_sigma)
    return None if k is None else int(record.steps[k - 1])


DIP_TEST = Detector("dip-test", "the run's own dip test fires (NVISION_DIP_P_FA), held for 3 steps", _dip_test)
SIGMA_DROP = Detector("sigma-drop", "the first steep, lasting drop of the claimed sigma", _sigma_drop)
DETECTORS: dict[str, Detector] = {d.name: d for d in (DIP_TEST, SIGMA_DROP)}


def compare_detectors(records: list[RunRecord], a: Detector = DIP_TEST, b: Detector = SIGMA_DROP) -> dict:
    """How far two detection definitions are apart on the same runs.

    Per run ``delta = step_b - step_a`` (measurements); medians and quartiles over the runs both detect, and the share
    each detects. The stored series of a simulated run is subsampled, so ``SIGMA_DROP`` there has the resolution of
    the stored steps.
    """
    pairs, only_a, only_b = [], 0, 0
    for r in records:
        sa, sb = a.step(r), b.step(r)
        if sa is not None and sb is not None:
            pairs.append((sa, sb))
        elif sa is not None:
            only_a += 1
        elif sb is not None:
            only_b += 1
    out: dict = {
        "a": a.name,
        "b": b.name,
        "n_runs": len(records),
        "n_both": len(pairs),
        "only_a": only_a,
        "only_b": only_b,
    }
    if pairs:
        sa = np.array([p[0] for p in pairs], dtype=np.float64)
        sb = np.array([p[1] for p in pairs], dtype=np.float64)
        d = sb - sa
        out.update(
            {
                "median_a": float(np.median(sa)),
                "median_b": float(np.median(sb)),
                "delta_median": float(np.median(d)),
                "delta_q25": float(np.percentile(d, 25)),
                "delta_q75": float(np.percentile(d, 75)),
                "ratio_median": float(np.median(sb / sa)),
            }
        )
    return out


def finite_or_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)
