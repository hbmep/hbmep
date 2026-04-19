import jax
import jax.numpy as jnp


def rectified_logistic(y, a, b, g, h, v):
    r"""
    Returns :math:`x` such that :math:`f(x) = y`, where :math:`f` is the rectified-logistic function.

    Notes
    -----
    - If :math:`y < g`, returns NaN.
    - If :math:`y = g`, returns :math:`a`.
    """
    y = jnp.where(y < g, jnp.nan, y)
    z = (jnp.log(h) - jnp.log(v)) / b
    z = a + z
    z = logistic4(y, z, b, g - v, h + v)
    return z


def logistic5(y, a, b, g, h, v):
    r"""
    Returns :math:`x` such that :math:`f(x) = y`, where :math:`f` is the logistic-5 function.
    """
    z = jnp.power((y - g) / h, v)
    z = jax.scipy.special.logit(z) + jnp.log(-1 + jnp.power(2, v))
    z = a + (z / b)
    return z


def logistic4(y, a, b, g, h):
    r"""
    Returns :math:`x` such that :math:`f(x) = y`, where :math:`f` is the logistic-4 function.
    """
    z = (y - g) / h
    z = jax.scipy.special.logit(z)
    z = a + (z / b)
    return z
