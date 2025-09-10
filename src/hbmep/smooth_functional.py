import jax
import jax.numpy as jnp

from hbmep import functional as F

EPS = 1e-3


def _smooth_max(x, eps):
    """
    Smooth approximation of the maximum function: x -> max(0, x)
    """
    eps = eps / jnp.log(2)
    z = x + (
        eps * jax.nn.softplus(-x / eps)
    )
    return z


def rectified_logistic(x, a, b, g, h, v, eps=EPS):
    """
    Smooth approximation of the rectified-logistic function
    """
    z = F._logistic_transform(x, a, b, h, v)
    z = _smooth_max(z, eps)
    return g + z


def rectified_linear(x, a, b, g, eps=EPS):
    """
    Smooth approximation of the rectified-linear function
    """
    z = F._linear_transform(x, a, b)
    z = _smooth_max(z, eps)
    return g + z


def rectified_logistic_inInflectionParam(x, a, b, g, h, v, eps=EPS):
    """
    Smooth approximation of rectified-logistic function in S50 parameterization
    """
    a = F.get_threshold(a, b, g, h, v)
    return rectified_logistic(x, a, b, g, h, v, eps)
