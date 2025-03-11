import jax
import jax.numpy as jnp


def _linear_transform(x, a, b):
    return jnp.multiply(b, x - a)


def _logistic_transform(x, a, b, h, v):
    z = _linear_transform(x, a, b) - jnp.log(h) + jnp.log(v)
    z = jax.nn.sigmoid(z)
    z = (h + v) * z
    z = -v + z
    return z


def _threshold_s50_delta(a, b, h, v):
    z = jnp.log(v) - jnp.log(h + 2 * v)
    return z / b


def rectified_logistic(x, a, b, g, h, v):
    """ Rectified-logistic function in threshold parameterization """
    z = _logistic_transform(x, a, b, h, v)
    z = jax.nn.relu(z)
    return g + z


def logistic5(x, a, b, g, h, v):
    """ Logistic-5 function """
    z = _linear_transform(x, a, b) - jnp.log(-1 + jnp.power(2, v))
    z = jax.nn.sigmoid(z)
    z = jnp.power(z, 1 / v)
    z = h * z
    return g + z


def logistic4(x, a, b, g, h):
    """ Logistic-4 function """
    z = _linear_transform(x, a, b)
    z = jax.nn.sigmoid(z)
    z = h * z
    return g + z


def rectified_linear(x, a, b, g):
    """ Rectified-linear function """
    z = _linear_transform(x, a, b)
    z = jax.nn.relu(z)
    return g + z


def threshold(a, b, g, h, v):
    """ Compute threshold of the rectified-logistic in S50 parameterization """
    return a + _threshold_s50_delta(a, b, h, v)


def s50(a, b, g, h, v):
    """ Compute S50 of the rectified-logistic in threshold parameterization """
    return a - _threshold_s50_delta(a, b, h, v)


def rectified_logistic_s50(x, a, b, g, h, v):
    """
    Rectified-logistic function in S50 parameterization
    """
    a = threshold(a, b, g, h, v)
    return rectified_logistic(x, a, b, g, h, v)


def grad(fn, x, *args):
    """ Compute the gradient of a function """
    args = jnp.broadcast_arrays(x, *args)
    grad_fn = jax.grad(fn)
    for _ in range(x.ndim):
        grad_fn = jax.vmap(grad_fn)
    return grad_fn(*args)
