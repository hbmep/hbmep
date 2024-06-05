import jax
from jax import jit
import jax.numpy as jnp


@jit
def _linear_transform(
    x, a, b
):
    return jnp.multiply(b, x - a)


@jit
def rectified_logistic(x, a, b, L, ell, H):
    """ This is the rectified logistic function """
    return (
        L
        + jax.nn.relu(
            - ell
            + jnp.multiply(
                H + ell,
                jax.nn.sigmoid(
                    _linear_transform(x, a, b)
                    - jnp.log(H / ell)
                )
            )
        )
    )


@jit
def logistic5(x, a, b, v, L, H):
    """ This is the logistic-5 function """
    return (
        L
        + jnp.multiply(
            H,
            jnp.power(
                jax.nn.sigmoid(
                    _linear_transform(x, a, b)
                    - jnp.log(-1 + jnp.power(2, v))
                ),
                jnp.true_divide(1, v)
            )
        )
    )


@jit
def logistic4(x, a, b, L, H):
    """ This is the logistic-4 function """
    return (
        L
        + jnp.multiply(
            H,
            jax.nn.sigmoid(_linear_transform(x, a, b))
        )
    )


@jit
def rectified_linear(x, a, b, L):
    """ This is the rectified linear function """
    return (
        L
        + jax.nn.relu(_linear_transform(x, a, b))
    )


def prime(fn, x, *args):
    """ Compute the gradient of a function """
    grad = jax.grad(fn, argnums=0)
    for _ in range(len(x.shape)):
        grad = jax.vmap(grad)
    return grad(x, *args)


def solve_rectified_logistic(y, a, b, L, ell, H):
    """
    This solves the rectified logistic function at y
    Use this function with y = L + (H / 2) to get the S50
    """
    return jnp.where(
        jnp.logical_and(y > L, y < L + H),
        a
        - jnp.true_divide(
            jnp.log(jnp.multiply(
                jnp.true_divide(ell, H),
                - 1
                + jnp.true_divide(
                    H + ell,
                    y + ell - L
                )
            )),
            b
        ),
        jnp.nan
    )


@jit
def rectified_logistic_s50(x, a, b, L, ell, H):
    """
    This is the rectified logistic function
    with S50 parameterization
    """
    return (
        L
        + jax.nn.relu(
            - ell
            + jnp.multiply(
                H + ell,
                jax.nn.sigmoid(
                    _linear_transform(x, a, b)
                    - jnp.log(
                        - 1
                        + jnp.true_divide(
                            jnp.multiply(2, H + ell),
                            H + jnp.multiply(2, ell)
                        )
                    )
                )
            )
        )
    )


def get_threshold_from_rectified_logistic_s50(a, b, ell, H):
    """
    This returns the threshold from the rectified logistic
    function with S50 parameterization
    """
    return (
        a
        + jnp.true_divide(
            jnp.log(jnp.multiply(
                jnp.true_divide(ell, H),
                - 1
                + jnp.true_divide(
                    jnp.multiply(2, H + ell),
                    H + jnp.multiply(2, ell)
                )
            )),
            b
        )
    )


@jit
def _rectified_logistic(x, a, b, v, L, ell, H):
    """ This is a rectified version of logistic-5 function """
    return (
        L
        + jax.nn.relu(
            - ell
            + jnp.multiply(
                H + ell,
                jnp.power(
                    jax.nn.sigmoid(
                        _linear_transform(x, a, b)
                        - jnp.log(
                            - 1
                            + jnp.power(
                                1 + jnp.true_divide(H, ell),
                                v
                            )
                        )
                    ),
                    jnp.true_divide(1, v)
                )
            )
        )
    )
