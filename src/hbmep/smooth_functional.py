import jax
import jax.numpy as jnp

from hbmep import functional as F


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


def rectified_logistic(x, a, b, L, ell, H, eps=1e-3):
    """
    Smooth approximation of rectified logistic function
    """
    z = F.linear_transform(x, a, b) - jnp.log(H) + jnp.log(ell)
    z = jax.nn.sigmoid(z)
    z = (H + ell) * z
    z = -ell + z
    z = smooth_max(z, eps)
    return L + z


def rectified_linear(x, a, b, L, eps=1e-3):
    """
    Smooth approximation of rectified linear function
    """
    z = F.linear_transform(x, a, b)
    z = smooth_max(z, eps)
    return L + z


def rectified_logistic_S50(x, a, b, L, ell, H, eps=1e-2):
    """
    This is the rectified logistic function
    in the S50 parameterization
    """
    z = F.linear_transform(x, a, b) - jnp.log(H) + jnp.log(H + 2 * ell)
    z = jax.nn.sigmoid(z)
    z = (H + ell) * z
    z = -ell + z
    z = smooth_max(z, eps)
    return L + z
