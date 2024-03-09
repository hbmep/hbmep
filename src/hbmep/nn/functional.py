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
    return (
        L
        + jnp.multiply(
            H,
            jax.nn.sigmoid(_linear_transform(x, a, b))
        )
    )


@jit
def rectified_linear(x, a, b, L):
    return (
        L
        + jax.nn.relu(_linear_transform(x, a, b))
    )


@jit
def _rectified_logistic5(x, a, b, v, L, ell, H):
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


@jit
def rectified_logistic_S50(x, a, b, L, ell, H):
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


def prime(fn, x, *args):
    grad = jax.grad(fn, argnums=0)
    for _ in range(len(x.shape)):
        grad = jax.vmap(grad)
    return grad(x, *args)
