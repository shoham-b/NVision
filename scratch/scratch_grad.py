import jax
import jax.numpy as jnp
from jax import grad, jacrev
import numpy as np

def nv_center_lorentzian_jax_new(
    x,
    frequency,
    linewidth,
    split,
    k_np,
    dip_depth,
    background,
):
    # Dimensionless coordinate and spacing
    # Add epsilon to prevent division by zero
    omega = jnp.where(linewidth > 1e-10, linewidth, 1e-10)
    x_dim = (x - frequency) / omega
    alpha = split / omega

    # Spin Population Conservation
    k = jnp.where(k_np > 1e-10, k_np, 1e-10)
    p_sum = k + 1.0 + (1.0 / k)

    # Individual Peak Heights
    c_total = dip_depth
    p_0 = c_total / p_sum
    p_L = c_total * (k / p_sum)
    p_R = c_total * ((1.0 / k) / p_sum)

    # Decoupled, Normalized Evaluation
    return background - (
        p_L / ((x_dim + alpha)**2 + 1.0) +
        p_0 / (x_dim**2 + 1.0) +
        p_R / ((x_dim - alpha)**2 + 1.0)
    )

def wrapper(params, x):
    frequency, linewidth, split, k_np, dip_depth = params
    return nv_center_lorentzian_jax_new(x, frequency, linewidth, split, k_np, dip_depth, 1.0)

jac_fn = jacrev(wrapper, argnums=0)

x = 2.87e9
params = jnp.array([2.87e9, 1e6, 2e6, 1.0, 0.5])
print(jac_fn(params, x))
