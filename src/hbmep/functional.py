import jax
import jax.numpy as jnp


def _linear_transform(x, a, b):
    return jnp.multiply(b, x - a)


def _logistic_transform(x, a, b, ell, H):
    z = _linear_transform(x, a, b) - jnp.log(H) + jnp.log(ell)
    z = jax.nn.sigmoid(z)
    z = (H + ell) * z
    z = -ell + z
    return z


def _threshold_s50_delta(a, b, ell, H):
    z = jnp.log(ell) - jnp.log(H + 2 * ell)
    return z / b


def rectified_logistic(x, a, b, L, ell, H):
    """ Rectified-logistic function in threshold parameterization """
    z = _logistic_transform(x, a, b, ell, H)
    z = jax.nn.relu(z)
    return L + z


def logistic5(x, a, b, v, L, H):
    """ Logistic-5 function """
    z = _linear_transform(x, a, b) - jnp.log(-1 + jnp.power(2, v))
    z = jax.nn.sigmoid(z)
    z = jnp.power(z, 1 / v)
    z = H * z
    return L + z


def logistic4(x, a, b, L, H):
    """ Logistic-4 function """
    z = _linear_transform(x, a, b)
    z = jax.nn.sigmoid(z)
    z = H * z
    return L + z


def rectified_linear(x, a, b, L):
    """ Rectified-linear function """
    z = _linear_transform(x, a, b)
    z = jax.nn.relu(z)
    return L + z


def threshold(a, b, L, ell, H):
    """ Compute threshold of the rectified-logistic in S50 parameterization """
    return a + _threshold_s50_delta(a, b, ell, H)


def s50(a, b, L, ell, H):
    """ Compute S50 of the rectified-logistic in threshold parameterization """
    return a - _threshold_s50_delta(a, b, ell, H)


def rectified_logistic_s50(x, a, b, L, ell, H):
    """
    Rectified-logistic function in S50 parameterization
    """
    a = threshold(a, b, L, ell, H)
    return rectified_logistic(x, a, b, L, ell, H)


def grad(fn, x, *args):
    """ Compute the gradient of a function """
    x = jnp.broadcast_to(x, args[0].shape)
    grad = jax.grad(fn, argnums=0)
    for _ in range(len(x.shape)):
        grad = jax.vmap(grad)
    return grad(x, *args)

