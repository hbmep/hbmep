import jax
from jax import jit
import jax.numpy as jnp


@jit
def relu(x, a, b, L):
    return (
        L
        + jnp.where(
            jnp.less(x, a),
            0.,
            jnp.multiply(b, x - a)
        )
    )


@jit
def logistic4(x, a, b, L, H):
    return (
        L + jnp.true_divide(
            H,
            1 + jnp.exp(jnp.multiply(-b, x - a))
        )
    )


@jit
def logistic5(x, a, b, v, L, H):
    return (
        L + jnp.true_divide(
            H,
            jnp.power(
                1 + jnp.multiply(
                    v,
                    jnp.exp(jnp.multiply(-b, x - a))
                ),
                jnp.true_divide(1, v)
            )
        )
    )


@jit
def rectified_logistic(x, a, b, v, L, ell, H):
    return (
        L
        + jnp.where(
            jnp.less(x, a),
            0.,
            -ell + jnp.true_divide(
                H + ell,
                jnp.power(
                    1
                    + jnp.multiply(
                        -1
                        + jnp.power(
                            jnp.true_divide(H + ell, ell),
                            v
                        ),
                        jnp.exp(jnp.multiply(-b, x - a))
                    ),
                    jnp.true_divide(1, v)
                )
            )
        )
    )


@jit
def prime(fn, x, *args):
    grad = jax.grad(fn, argnums=0)
    for _ in range(len(x.shape)):
        grad = jax.vmap(grad)
    return grad(x, *args)
