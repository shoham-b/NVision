# NVision — Equations and Approximations

Mathematical reference for the **additive Gaussian measurement-noise** path.  Split into:

- **[sbed_and_smc.md](sbed_and_smc.md)** — the SMC particle filter (`smc_marginal.py`), its unit-cube extension (`unit_cube_smc_marginal.py`), the SBED acquisition locator (`sbed_locator.py`), the Gaussian Fisher information / CRLB, and the convergence criteria. These form one tightly-coupled inference stack and share the same symbols and constants.
- **[metrics.md](metrics.md)** — per-run evaluation metrics: point-estimate error, the uniform-sampling baseline, and failure classification (`metrics.py`).

> **Scope:** only the Gaussian-noise path is covered. The Rao-Blackwell (Inverse-Gamma) noise marginalization, Poisson-count likelihoods, and non-SBED locators are intentionally out of scope.
>
> **On "sweep":** there is no standalone sweep locator in this scope. The only sweep-related math is the *uniform-sampling baseline* used to benchmark SBED's step count — it lives in [metrics.md](metrics.md) §2 because it is a comparison metric, not a locator.
