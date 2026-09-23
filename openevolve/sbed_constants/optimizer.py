"""Noise-aware Bayesian optimization over the 7 SBED acquisition constants.

Why this is not "run N candidates and keep the best"
----------------------------------------------------

The LLM-driven search this replaces failed for a statistical reason, not an
implementation one. It promoted ``argmax`` over ~30 noisy evaluations, and
``max`` of noisy estimates is a biased estimator of the underlying value: with
evaluation sd ~0.06--0.08, the expected maximum over 30 draws sits roughly 2 sd
above the mean *even when every candidate is functionally identical to the
baseline*. Both of its "winners" were later re-measured against the unmodified
baseline and lost (0.5855 baseline vs 0.5096 and 0.4974). The apparent wins were
exactly the size the winner's curse predicts.

So every design choice here is aimed at that one failure mode:

1.  **The noise is modelled explicitly.** The surrogate is a Gaussian process
    whose kernel carries a ``WhiteKernel`` nugget, fitted by marginal
    likelihood. The run therefore *measures* the evaluation noise instead of
    assuming it, and prints it -- if the fitted noise sd comes back at 0.07 and
    the spread of posterior means across the whole box is 0.02, the honest
    conclusion is "these constants do not matter", and the run will say so.

2.  **The recommendation is the argmax of the posterior mean, never the argmax
    of an observed score.** The posterior mean shrinks each observation toward
    what its neighbours say, which is precisely the correction the winner's
    curse needs. A point that scored 0.68 once, surrounded by points scoring
    0.55, is reported at roughly 0.56 -- not 0.68.

3.  **Incumbents are re-evaluated.** Every ``reeval_every`` iterations the loop
    spends its evaluation on the *current recommendation* rather than on a new
    proposal. Replicates at the same coordinates are what let the GP separate
    the nugget from the signal at all; without them the marginal likelihood can
    explain everything with a short length scale and near-zero noise.

4.  **The acquisition uses the latent (noise-free) standard deviation.** With a
    ``WhiteKernel`` in the kernel, ``predict(return_std=True)`` returns a std
    that includes observation noise, which never falls below the nugget -- so
    expected improvement would stay large everywhere forever and the search
    would degenerate into random sampling. The nugget is subtracted before the
    acquisition is computed.

5.  **Expected improvement is taken against the posterior-mean incumbent**, not
    against ``max(y)``. Chasing a lucky draw is the same error as reporting one.

CMA-ES was the other candidate design and would also have been defensible (it
is rank-based and tolerates noise well). Bayesian optimization won on budget:
at ~35 s per evaluation a realistic run is 100--200 evaluations, and 7-dimensional
CMA-ES with a population of ~10 gets 10--20 generations out of that, which is
too few for the covariance adaptation to pay off. A GP also gives something
CMA-ES does not: a directly reportable estimate of the observation noise and of
how much of the score range is real.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .candidate import check_baseline_render, check_template
from .objective import default_grid, evaluate_params
from .space import BASELINE_CUBE, DIM_NAMES, N_DIMS, format_params, from_cube, repair_cube

# Prior expectation for the observation noise, in standardised-score units. If
# the constants turn out not to matter, essentially all of the score variance is
# noise, so 1.0 is the honest starting guess and the marginal likelihood moves it
# from there. Starting near zero instead biases the first few fits toward
# believing every wiggle is signal.
_NOISE_INIT = 0.7
_NOISE_BOUNDS = (1e-3, 1e1)
_LENGTH_SCALE_BOUNDS = (5e-2, 1e2)


@dataclass
class Observation:
    """One completed evaluation."""

    cube: np.ndarray
    score: float
    params: dict[str, float]
    kind: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "score": self.score,
            "cube": [float(v) for v in self.cube],
            "params": {k: float(v) for k, v in self.params.items()},
            "metrics": {k: _json_safe(v) for k, v in self.metrics.items()},
        }


def _json_safe(value):
    """JSON-encodable form of a metric value.

    Metrics are not all floats: ``median_steps`` is NaN when nothing converged,
    and the harness's failure path attaches a string ``error``. Calling
    ``np.isfinite`` on either would raise *while writing the log*, losing the
    evaluation that had just been paid for.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value) if np.isfinite(value) else None
    return str(value)


def _build_gp(n_dims: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.full(n_dims, 0.5), length_scale_bounds=_LENGTH_SCALE_BOUNDS, nu=2.5
    ) + WhiteKernel(noise_level=_NOISE_INIT, noise_level_bounds=_NOISE_BOUNDS)
    # alpha stays at its numerical-jitter default: the observation noise is the
    # WhiteKernel's job, and doubling it up here would make the fitted nugget --
    # the number this whole design is built to report -- an underestimate.
    return GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=8, alpha=1e-10)


def _fitted_noise_variance(gp: GaussianProcessRegressor) -> float:
    """The fitted WhiteKernel nugget, in the units the GP was fitted in."""
    total = 0.0
    for name, value in gp.kernel_.get_params().items():
        if name.endswith("noise_level") and isinstance(value, int | float):
            total += float(value)
    return total


class NoiseAwareBO:
    """GP-based optimizer with an explicit observation-noise model."""

    def __init__(self, seed: int = 0, n_candidates: int = 4096) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates
        self.observations: list[Observation] = []
        self._gp: GaussianProcessRegressor | None = None
        self._y_mean = 0.0
        self._y_scale = 1.0

    # ---------------------------------------------------------------- fitting

    @property
    def X(self) -> np.ndarray:  # conventional design-matrix name
        return np.array([o.cube for o in self.observations], dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.array([o.score for o in self.observations], dtype=float)

    def fit(self) -> None:
        """Refit the surrogate. Standardisation is done here, not by sklearn.

        ``normalize_y=True`` would leave the fitted nugget in an internal scale
        that has to be converted back through a private attribute; doing the
        standardisation explicitly keeps the nugget, the predictive std and the
        acquisition all in one consistent scale that can be reported honestly.
        """
        y = self.y
        if y.size < 3:
            self._gp = None
            return
        self._y_mean = float(np.mean(y))
        scale = float(np.std(y))
        self._y_scale = scale if scale > 1e-9 else 1.0
        gp = _build_gp(N_DIMS)
        with warnings.catch_warnings():
            # A length scale pinned at its upper bound is the expected, useful
            # answer here -- it says "this constant does not matter" -- and a
            # nugget at a bound is normal while the design is still small. Both
            # raise ConvergenceWarning on every refit, which would bury the
            # per-iteration score lines under hundreds of lines of noise.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp.fit(self.X, (y - self._y_mean) / self._y_scale)
        self._gp = gp

    def noise_sd(self) -> float | None:
        """Fitted observation-noise sd, in original score units."""
        if self._gp is None:
            return None
        return float(np.sqrt(max(_fitted_noise_variance(self._gp), 0.0)) * self._y_scale)

    def _predict(self, cubes: np.ndarray, latent: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and sd in original score units.

        ``latent=True`` subtracts the fitted nugget from the predictive variance,
        giving the uncertainty about the *underlying* value rather than about a
        future single noisy measurement. Expected improvement must use the
        former: with the nugget included the sd has a floor it can never go
        below, so no amount of sampling ever makes a region look settled and the
        search stops converging.
        """
        if self._gp is None:
            raise RuntimeError("surrogate is not fitted yet")
        mean, sd = self._gp.predict(np.atleast_2d(cubes), return_std=True)
        var = np.square(sd)
        if latent:
            var = np.maximum(var - _fitted_noise_variance(self._gp), 0.0)
        return mean * self._y_scale + self._y_mean, np.sqrt(var) * self._y_scale

    # ------------------------------------------------------------ acquisition

    def _incumbent(self) -> float:
        """Best *posterior mean* among evaluated points -- not ``max(y)``."""
        mean, _ = self._predict(self.X, latent=True)
        return float(np.max(mean))

    def _expected_improvement(self, cubes: np.ndarray, xi: float = 0.005) -> np.ndarray:
        mean, sd = self._predict(cubes, latent=True)
        improvement = mean - self._incumbent() - xi
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 1e-12, improvement / np.maximum(sd, 1e-12), 0.0)
            ei = improvement * norm.cdf(z) + sd * norm.pdf(z)
        return np.where(sd > 1e-12, np.maximum(ei, 0.0), 0.0)

    def propose(self) -> np.ndarray:
        """Next cube point to evaluate."""
        if self._gp is None:
            return repair_cube(self.rng.random(N_DIMS))
        sampler = qmc.Sobol(d=N_DIMS, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        cubes = sampler.random(self.n_candidates)
        cubes = np.array([repair_cube(c) for c in cubes])
        ei = self._expected_improvement(cubes)
        order = np.argsort(ei)[::-1]

        best_cube, best_ei = cubes[order[0]], float(ei[order[0]])
        # Polish the top few Sobol hits. The repair projection makes the objective
        # piecewise (min_obs is integral), so this is a local refinement of an
        # already-good point, not the search itself.
        for index in order[:5]:
            result = minimize(
                lambda c: -float(self._expected_improvement(repair_cube(c)[None, :])[0]),
                cubes[index],
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * N_DIMS,
                options={"maxiter": 60},
            )
            if result.success and -float(result.fun) > best_ei:
                best_cube, best_ei = repair_cube(result.x), -float(result.fun)
        return best_cube

    # -------------------------------------------------------- recommendation

    def recommend(self) -> tuple[np.ndarray, float, float]:
        """(cube, posterior mean, posterior sd) of the best evaluated point.

        Restricted to points that were actually evaluated. Recommending the
        global argmax of the posterior mean over the whole box would hand back a
        location nothing was ever measured at, whose "score" is pure
        extrapolation -- a second way to manufacture an optimistic number.
        """
        if not self.observations:
            raise RuntimeError("nothing evaluated yet")
        if self._gp is None:
            index = int(np.argmax(self.y))
            return self.observations[index].cube, float(self.y[index]), float("nan")
        mean, sd = self._predict(self.X, latent=True)
        index = int(np.argmax(mean))
        return self.X[index], float(mean[index]), float(sd[index])

    # ---------------------------------------------------------------- record

    def tell(self, cube: np.ndarray, metrics: dict[str, float], kind: str) -> Observation:
        params = from_cube(cube)
        score = float(metrics.get("combined_score", 0.0) or 0.0)
        obs = Observation(
            cube=np.asarray(cube, dtype=float),
            score=score if np.isfinite(score) else 0.0,
            params=params,
            kind=kind,
            metrics=metrics,
        )
        self.observations.append(obs)
        return obs


def run(
    n_init: int = 16,
    n_iter: int = 84,
    reeval_every: int = 4,
    seed: int = 0,
    repeats: int | None = None,
    grid: list[tuple[str, str, int]] | None = None,
    crn_seed: int | None = None,
    log_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    """Run the optimization loop. Returns a summary dict."""
    # Both guards are cheap and both are worth having before spending an hour:
    # the first proves every constant site still matches and the rendered module
    # compiles, the second proves rendering the defaults reproduces the baseline
    # algorithm rather than some drifted variant of it.
    check_template()
    check_baseline_render()
    grid = list(grid) if grid is not None else default_grid()
    if repeats is not None:
        grid = [(generator, noise, repeats) for generator, noise, _ in grid]

    opt = NoiseAwareBO(seed=seed)
    log_file = Path(log_path) if log_path is not None else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if resume and log_file.exists():
            for line in log_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                opt.tell(np.array(record["cube"], dtype=float), record.get("metrics") or {}, record.get("kind", "init"))
                opt.observations[-1].score = float(record["score"])
            opt.fit()
            print(f"[resume] restored {len(opt.observations)} observations from {log_file}")

    def _record(obs: Observation) -> None:
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(obs.to_json()) + "\n")

    def _evaluate(cube: np.ndarray) -> dict[str, float]:
        """Evaluate a point, turning a crash into a zero rather than a lost run.

        A candidate can legitimately raise -- the acquisition functions fail fast
        on out-of-domain frequencies rather than clamping them, which is exactly
        how an unreasonable constant should behave. The harness's own evaluator
        scores such a candidate 0 and moves on; letting the exception escape here
        would instead throw away every evaluation already paid for in this run.
        Recording it as 0 also teaches the surrogate to avoid that region.
        """
        try:
            return evaluate_params(from_cube(cube), grid=grid, crn_seed=crn_seed)
        except Exception as exc:  # a bad candidate must not end the run
            print(f"       evaluation failed, scoring 0: {type(exc).__name__}: {exc}")
            return {"combined_score": 0.0, "error": f"{type(exc).__name__}: {exc}"[:500]}

    # The baseline is always the first point in the design. The whole run is a
    # comparison against it, so it must be inside the surrogate's data rather
    # than a remembered number from a previous session on a different machine.
    pending: list[tuple[np.ndarray, str]] = []
    if not opt.observations:
        pending.append((BASELINE_CUBE.copy(), "baseline"))
        sampler = qmc.Sobol(d=N_DIMS, scramble=True, seed=seed)
        for cube in sampler.random(max(0, n_init - 1)):
            pending.append((repair_cube(cube), "init"))

    started = time.perf_counter()
    for cube, kind in pending:
        obs = opt.tell(cube, _evaluate(cube), kind)
        _record(obs)
        print(f"[{len(opt.observations):3d}] {kind:8s} score={obs.score:.4f}  {format_params(obs.params)}")
    opt.fit()

    for iteration in range(1, n_iter + 1):
        if opt._gp is not None and reeval_every > 0 and iteration % reeval_every == 0:
            cube, _, _ = opt.recommend()
            kind = "reeval"
        else:
            cube = opt.propose()
            kind = "ei"
        obs = opt.tell(cube, _evaluate(cube), kind)
        _record(obs)
        opt.fit()
        noise = opt.noise_sd()
        noise_text = f"{noise:.4f}" if noise is not None else "n/a"
        print(
            f"[{len(opt.observations):3d}] {kind:8s} score={obs.score:.4f}  "
            f"noise_sd={noise_text}  {format_params(obs.params)}"
        )

    elapsed = time.perf_counter() - started
    best_cube, best_mean, best_sd = opt.recommend()
    best_params = from_cube(best_cube)
    baseline = [o for o in opt.observations if np.allclose(o.cube, BASELINE_CUBE, atol=1e-9)]
    baseline_scores = [o.score for o in baseline]

    summary = {
        "n_observations": len(opt.observations),
        "elapsed_s": elapsed,
        "noise_sd": opt.noise_sd(),
        "best_params": best_params,
        "best_posterior_mean": best_mean,
        "best_posterior_sd": best_sd,
        "best_observed_score": float(np.max(opt.y)) if opt.observations else float("nan"),
        "baseline_n": len(baseline_scores),
        "baseline_observed_mean": float(np.mean(baseline_scores)) if baseline_scores else float("nan"),
        "dim_names": list(DIM_NAMES),
    }
    return summary
