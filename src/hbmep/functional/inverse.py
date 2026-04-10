import jax
import jax.numpy as jnp


def rectified_logistic(y, a, b, g, h, v):
    r"""
    Inverse of rectified-logistic function
    """
    y = jnp.where(y < g, jnp.nan, y)
    z = (jnp.log(h) - jnp.log(v)) / b
    z = a + z
    z = logistic4(y, z, b, g - v, h + v)
    return z


def logistic5(y, a, b, g, h, v):
    r"""
    Inverse of logistic-5 function
    """
    z = jnp.power((y - g) / h, v)
    z = jax.scipy.special.logit(z) + jnp.log(-1 + jnp.power(2, v))
    z = a + (z / b)
    return z


def logistic4(y, a, b, g, h):
    r"""
    Inverse of logistic-4 function
    """
    z = (y - g) / h
    z = jax.scipy.special.logit(z)
    z = a + (z / b)
    return z
