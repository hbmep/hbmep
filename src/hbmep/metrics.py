from jax import numpy as jnp, vmap


def lp_risk(true, samples, *, p=1, axis=-1, root=False, ignore_nan=True):
    """
    Compute posterior expected Lp risk error.

    Parameters
    ----------
    true : Array
        True values, broadcast-compatible with `samples`, with posterior
        sample axis either present as size 1 or absent.
    samples: Array
        Posterior samples. `axis` is the posterior sample axis.
    p : int | float
        Power of the Lp loss.
    axis : int
        Posterior sample axis.
    root : bool
        If False, computes :math:`\mathbb{E}[|x - y|^p]`,
        otherwise :math:`\mathbb{E}[|x - y|^p]^{1/p}`.
    ignore_nan : bool
        If True, use `jnp.nanmean`; otherwise use `jnp.mean`.

    Returns
    -------
    Array
        Posterior expected Lp risk with the posterior sample axis removed.

    Notes
    -----
    - p=1, root=False: L1 risk / MAE
    - p=2, root=False: L2 risk / MSE
    - p=2, root=True: RMSE
    """
    samples = jnp.asarray(samples)
    true = jnp.asarray(true)

    axis = axis % samples.ndim

    if true.ndim == samples.ndim - 1:
        true = jnp.expand_dims(true, axis)  # add dimension 1 to the left of axis

    err = jnp.abs(samples - true) ** p
    mean_fn = jnp.nanmean if ignore_nan else jnp.mean
    out = mean_fn(err, axis=axis)

    if root:
        out = out ** (1 / p)

    return out


def crps_unbiased(true, samples, *, axis=-1, batch_size=256):
    """
    Unbiased Monte Carlo estimator of CRPS.

    Parameters
    ----------
    true : Array
        True values, broadcast-compatible with `samples`, with posterior
        sample axis either present as size 1 or absent.
    samples : Array
        Posterior samples. `axis` is the posterior sample axis.
    axis : int
        Posterior sample axis.
    batch_size : int
        Number of non-posterior elements processed per batch. Larger values use
        more memory.

    Returns
    -------
    Array
        CRPS with posterior sample axis removed.

    Notes
    -----
    - Computes :math:`\mathbb{E}[|X - y|] - \\frac{1}{2}\mathbb{E}[|X - X'|]`
      using the unbiased finite-sample estimator.
    """
    samples = jnp.asarray(samples)
    true = jnp.asarray(true)

    axis = axis % samples.ndim

    if true.ndim == samples.ndim - 1:
        true = jnp.expand_dims(true, axis)  # add dimension 1 to the left of axis

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
