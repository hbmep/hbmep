import jax.numpy as jnp
from jax import vmap

from hbmep.device import execute_on_gpu


def l1_risk(true, samples, *, axis=-1):
    """
    Compute posterior expected absolute error.

    true:
        True values, broadcast-compatible with `samples`, with posterior
        sample axis either present as size 1 or absent.

    samples:
        Posterior samples. `axis` is the posterior sample axis.

    returns:
        Mean absolute posterior error with posterior sample axis removed.
    """
    samples = jnp.asarray(samples)
    true = jnp.asarray(true)

    axis = axis % samples.ndim

    if true.ndim == samples.ndim - 1:
        true = jnp.expand_dims(true, axis)

    return jnp.mean(
        jnp.abs(samples - true),
        axis=axis,
    )


@execute_on_gpu
def crps_unbiased(true, samples, *, axis=-1, batch_size=256):
    """
    true:
        True values, broadcast-compatible with `samples`, with posterior
        sample axis either present as size 1 or absent.

    samples:
        Posterior samples. `axis` is the posterior sample axis.

    returns:
        CRPS with posterior sample axis removed.
    """
    samples = jnp.asarray(samples)
    true = jnp.asarray(true)

    axis = axis % samples.ndim

    if true.ndim == samples.ndim - 1:
        true = jnp.expand_dims(true, axis)

    true = jnp.broadcast_to(true, samples.shape)

    samples = jnp.moveaxis(samples, axis, -1)
    true = jnp.moveaxis(true, axis, -1)

    out_shape = samples.shape[:-1]
    S = samples.shape[-1]

    samples_flat = samples.reshape((-1, S))
    true_flat = true[..., 0].reshape((-1,))

    N = samples_flat.shape[0]

    def crps_unbiased_one(true, samples):
        """
        true : ()
        samples : (S,)
        """
        S = samples.shape[0]
        if S < 2:
            raise ValueError("Need at least 2 posterior samples")
        term1 = jnp.mean(jnp.abs(samples - true))
        diff = jnp.abs(
            samples[:, None]
            - samples[None, :]
        )
        pairwise_sum = jnp.sum(diff)
        term2 = 0.5 * pairwise_sum / (S * (S - 1))
        return term1 - term2

    vals = []
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        vals.append(
            vmap(crps_unbiased_one)(
                true_flat[start:stop],
                samples_flat[start:stop],
            )
        )

    return jnp.concatenate(vals).reshape(out_shape)
