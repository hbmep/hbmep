import jax
import jax.numpy as jnp
from jax.scipy.special import logit


def linear_transform(x, a, b):
    return jnp.multiply(b, x - a)


def rectified_logistic(x, a, b, L, ell, H):
    """ This is the rectified logistic function """
    return (
        L
        + jax.nn.relu(
            - ell
            + jnp.multiply(
                H + ell,
                jax.nn.sigmoid(
                    linear_transform(x, a, b)
                    - jnp.log(H / ell)
                )
            )
        )
    )


def logistic5(x, a, b, v, L, H):
    """ This is the logistic-5 function """
    return (
        L
        + jnp.multiply(
            H,
            jnp.power(
                jax.nn.sigmoid(
                    linear_transform(x, a, b)
                    - jnp.log(-1 + jnp.power(2, v))
                ),
                jnp.true_divide(1, v)
            )
        )
    )


def logistic4(x, a, b, L, H):
    """ This is the logistic-4 function """
    return (
        L
        + jnp.multiply(
            H,
            jax.nn.sigmoid(linear_transform(x, a, b))
        )
    )


def rectified_linear(x, a, b, L):
    """ This is the rectified linear function """
    return (
        L
        + jax.nn.relu(linear_transform(x, a, b))
    )


def prime(fn, x, *args):
    """ Compute the gradient of a function """
    x = jnp.broadcast_to(x, args[0].shape)
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
        + jnp.true_divide(
            logit(jnp.true_divide(y - L + ell, H + ell))
            + jnp.log(jnp.true_divide(H, ell)),
            b
        ),
        jnp.nan
    )


def rectified_logistic_S50(x, a, b, L, ell, H):
    """
    This is the rectified logistic function
    in the S50 parameterization
    """
    return (
        L
        + jax.nn.relu(
            - ell
            + jnp.multiply(
                H + ell,
                jax.nn.sigmoid(
                    linear_transform(x, a, b)
                    - jnp.log(jnp.true_divide(
                        H, H + jnp.multiply(2, ell)
                    ))
                )
            )
        )
    )
