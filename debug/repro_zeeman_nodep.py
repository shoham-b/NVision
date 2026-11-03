"""Print why primary_sensitivity_sq_phys is non-positive at the end of a run."""

import sys
import numpy as np
import nvision.metrics.stages as st
from nvision.belief.smc_marginal import SMCMarginalDistribution as B

orig = B.primary_sensitivity_sq_phys


def wrap(self, primary, x):
    g2 = orig(self, primary, x)
    if not float(np.max(g2)) > 0.0:
        est = self._estimates_unit()
        print(
            "ZERO/NaN g2:",
            "max",
            np.max(g2),
            "nan",
            bool(np.isnan(g2).any()),
            "est_unit",
            {k: round(v, 4) for k, v in est.items()},
            "bounds",
            self.physical_param_bounds.get(primary),
            flush=True,
        )
        j = self._param_names.index(primary)
        lo, hi = self.drive_freq_bounds_phys
        xs = (np.asarray(x) - lo) / (hi - lo)
        print("  drive bounds", (lo, hi), "fixed_drive", self.model.fixed_drive, "names", self._param_names, flush=True)
        for zu in (0.0, 0.2, 0.5, 0.74, 1.0):
            cols = [np.array([est[n]], dtype=self._particles.dtype) for n in self._param_names]
            cols[j] = np.array([zu], dtype=cols[j].dtype)
            y = np.asarray(self._predict_at_drives(xs, cols, [self.model.fixed_drive] * xs.size))[:, 0]
            if zu == 0.5:
                print("  particle dtype", self._particles.dtype, "y dtype", y.dtype, flush=True)
                for h in (1e-4, 1e-3, 1e-2):
                    ys = []
                    for sgn in (1, -1):
                        c2 = list(cols)
                        c2[j] = np.array([0.5 + sgn * h], dtype=self._particles.dtype)
                        ys.append(
                            np.asarray(
                                self._predict_at_drives(xs, c2, [self.model.fixed_drive] * xs.size), dtype=np.float64
                            )[:, 0]
                        )
                    print("  h", h, "max|dy|", float(np.max(np.abs(ys[0] - ys[1]))), flush=True)
            print("  zu", zu, "y min/max", float(y.min()), float(y.max()), "n_unique", len(np.unique(y)), flush=True)
    return g2


B.primary_sensitivity_sq_phys = wrap
from nvision.__main__ import main

sys.argv = ["nv"] + sys.argv[1:]
main()
