import jax
import jax.numpy as jnp


def linear_transform(x, a, b):
    return jnp.multiply(b, x - a)


def logistic_transform(x, a, b, h, v):
    z = linear_transform(x, a, b) - jnp.log(h) + jnp.log(v)
    z = jax.nn.sigmoid(z)
    z = (h + v) * z
    z = -v + z
    return z


def threshold_inflection_delta(a, b, h, v):
    z = jnp.log(v) - jnp.log(h + 2 * v)
    return z / b


def get_threshold(a, b, g, h, v):
    r"""
    Compute threshold of rectified-logistic in inflection parameterization
    """
    return a + threshold_inflection_delta(a, b, h, v)


def get_inflection(a, b, g, h, v):
    r"""
    Compute inflection of rectified-logistic in threshold parameterization
    """
    return a - threshold_inflection_delta(a, b, h, v)


def smooth_max(x, eps):
    r"""
    Smooth approximation of the maximum function: x -> max(0, x)
    """
    eps = eps / jnp.log(2)
    z = x + (
        eps * jax.nn.softplus(-x / eps)
    )
    return z
