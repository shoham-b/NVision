import numpy as np
from nvision.belief.smc_marginal import _weighted_mean_variance_1d

np.random.seed(0)
x = np.random.rand(100)
w = np.random.rand(100)

mean, var = _weighted_mean_variance_1d(x, w)

w_norm = w / w.sum()
var_np = np.average((x - np.average(x, weights=w_norm))**2, weights=w_norm)

print(f"Jit Var: {var}")
print(f"NP Var:  {var_np}")
assert np.isclose(var, var_np)
