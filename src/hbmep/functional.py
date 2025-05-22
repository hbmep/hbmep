import jax
import jax.numpy as jnp
import numpy as np


def with_attributes(**kwargs):
    """
    Decorator to add arbitrary attributes to a function.
    
    Example:
        @with_attributes(
            inverse=logistic4_inverse,
            derivative=logistic4_derivative
        )
        def logistic4(x, a, b, g, h):
            return ...
    """
    def decorator(func):
        for attr_name, attr_func in kwargs.items():
            setattr(func, attr_name, attr_func)
        return func
    return decorator


def _linear_transform(x, a, b):
    return jnp.multiply(b, x - a)


def _logistic_transform(x, a, b, h, v):
    z = _linear_transform(x, a, b) - jnp.log(h) + jnp.log(v)
    z = jax.nn.sigmoid(z)
    z = (h + v) * z
    z = -v + z
    return z


def _threshold_inflection_delta(a, b, h, v):
    z = jnp.log(v) - jnp.log(h + 2 * v)
    return z / b


def inverse_rectified_logistic(y, a, b, g, h, v):
    """ Inverse of rectified-logistic function """
    y = jnp.where(y < g, jnp.nan, y)
    z = (jnp.log(h) - jnp.log(v)) / b
    z = a + z
    z = logistic4.inverse(y, z, b, g - v, h + v)
    return z


def integrate_rectified_logistic(x, a, b, g, h, v):
    """ Closed form integral of rectified-logistic function """

    def body_integrate(x):
        z = b * (x - a) - jnp.log(h) + jnp.log(v)
        z = jnp.log(1 + jnp.exp(z))
        z *= (h + v) / b
        z += (g - v) * x
        return z


    z = body_integrate(jnp.maximum(x, a)) - body_integrate(a)
    z += g * jnp.minimum(x, a)
    # z = jnp.where(x <= a, g * x, (g * a) + body_integrate(x) - body_integrate(a))
    return z


def integrate_rectified_logistic_around_threshold(x, a, b, g, h, v):
    """
    Closed form integral of rectified-logistic function
    from a to (a + x), where x > 0
    """
    assert np.nanmin(x) > 0
    z = (h / v) + jnp.exp(b * x)
    z /= 1 + (h / v)
    z = jnp.log(z) * ((h + v) / b)
    z += (g - v) * x
    return z


@with_attributes(
    inverse=inverse_rectified_logistic,
    integrate=integrate_rectified_logistic,
    integrate_around_threshold=integrate_rectified_logistic_around_threshold
)
def rectified_logistic(x, a, b, g, h, v):
    """ Rectified-logistic function in threshold parameterization """
    z = _logistic_transform(x, a, b, h, v)
    z = jax.nn.relu(z)
    return g + z


def inverse_logistic5(y, a, b, g, h, v):
    """ Inverse of logistic-5 function """
    z = jnp.power((y - g) / h, v)
    z = jax.scipy.special.logit(z) + jnp.log(-1 + jnp.power(2, v))
    z = a + (z / b)
    return z


@with_attributes(inverse=inverse_logistic5)
def logistic5(x, a, b, g, h, v):
    """ Logistic-5 function """
    z = _linear_transform(x, a, b) - jnp.log(-1 + jnp.power(2, v))
    z = jax.nn.sigmoid(z)
    z = jnp.power(z, 1 / v)
    z = h * z
    return g + z


def inverse_logistic4(y, a, b, g, h):
    """ Inverse of logistic-4 function """
    z = (y - g) / h
    z = jax.scipy.special.logit(z)
    z = a + (z / b)
    return z


@with_attributes(inverse=inverse_logistic4)
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


def get_threshold(a, b, g, h, v):
    """ Compute threshold of rectified-logistic in inflection parameterization """
    return a + _threshold_inflection_delta(a, b, h, v)


def get_inflection(a, b, g, h, v):
    """ Compute inflection of rectified-logistic in threshold parameterization """
    return a - _threshold_inflection_delta(a, b, h, v)


def rectified_logistic_inInflectionParam(x, a, b, g, h, v):
    """
    Rectified-logistic function in inflection parameterization
    """
    a = get_threshold(a, b, g, h, v)
    return rectified_logistic(x, a, b, g, h, v)


def grad(fn, x, *args):
    """ Compute the gradient of a function """
    args = jnp.broadcast_arrays(x, *args)
    grad_fn = jax.grad(fn)
    for _ in range(x.ndim):
        grad_fn = jax.vmap(grad_fn)
    return grad_fn(*args)
