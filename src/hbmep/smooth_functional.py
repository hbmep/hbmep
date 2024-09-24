import jax
import jax.numpy as jnp

from hbmep import functional as F


# def smooth_max(x, eps):
#     """
#     Smooth approximation of the function
#     x \mapsto max(0, x)
#     """
#     z = jnp.log(
#         2 * jnp.cosh(x / eps)
#     )
#     z = eps * z
#     z = (1 / 2) * (x + z)
#     return z


# def smooth_max(x, eps):
#     """
#     Smooth approximation of the function
#     x \mapsto max(0, x)
#     """
#     eps = eps * 2
#     z = x + (
#         eps * jnp.log1p(jnp.exp(-x / eps))
#     )
#     return z


def smooth_max(x, eps):
    """
    Smooth approximation of the function
    x \mapsto max(0, x)
    """
    eps = eps / jnp.log(2)
    z = x + (
        eps * jax.nn.softplus(-x / eps)
    )
    return z


# def smooth_max(x, eps):
#     """
#     Smooth approximation of the function
#     x \mapsto max(0, x)
#     """
#     eps = eps * 2
#     z = eps * jnp.log1p(jnp.exp(x / eps))
#     return z


def rectified_logistic(x, a, b, L, ell, H, eps=1e-2):
    """
    Smooth approximation of rectified logistic function
    """
    z = F.linear_transform(x, a, b) - jnp.log(H) + jnp.log(ell)
    z = jax.nn.sigmoid(z)
    z = (H + ell) * z
    z = -ell + z
    z = smooth_max(z, eps)
    return L + z
