from jax import numpy as jnp, vmap


def lp_risk(true, samples, *, weights=None, p=1, axis=-1, root=False, ignore_nan=True):
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

    if weights is None:
        mean_fn = jnp.nanmean if ignore_nan else jnp.mean
        out = mean_fn(err, axis=axis)
    else:
        weights = jnp.asarray(weights)
        weights = weights / jnp.sum(weights)

        shape = [1] * samples.ndim
        shape[axis] = weights.shape[0]
        weights = weights.reshape(shape)

        if ignore_nan:
            mask = ~jnp.isnan(err)
            num = jnp.nansum(weights * err, axis=axis)
            den = jnp.sum(jnp.where(mask, weights, 0.0), axis=axis)
            out = num / den
        else:
            out = jnp.sum(weights * err, axis=axis)

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


def wasserstein1(samples1, samples2, *, weights1=None, weights2=None):
    """
    Weighted 1D Wasserstein-1 distance between two empirical distributions.

    Parameters
    ----------
    samples1, samples2 : Array
        Samples representing two empirical distributions.
    weights1, weights2 : Array, optional
        Nonnegative weights associated with `samples1`
        and `samples2`. If omitted, uniform weights are used.

    Returns
    -------
    Array
        Wasserstein-1 distance.
    """
    samples1 = jnp.asarray(samples1)
    samples2 = jnp.asarray(samples2)

    if weights1 is None:
        weights1 = jnp.ones_like(samples1) / samples1.size
    else:
        weights1 = jnp.asarray(weights1)
        weights1 = weights1 / jnp.sum(weights1)

    if weights2 is None:
        weights2 = jnp.ones_like(samples2) / samples2.size
    else:
        weights2 = jnp.asarray(weights2)
        weights2 = weights2 / jnp.sum(weights2)

    idx1 = jnp.argsort(samples1)
    idx2 = jnp.argsort(samples2)

    samples1 = samples1[idx1]
    samples2 = samples2[idx2]

    weights1 = weights1[idx1]
    weights2 = weights2[idx2]

    z = jnp.sort(jnp.concatenate([samples1, samples2]))
    dz = jnp.diff(z)

    cdf1 = jnp.cumsum(weights1)
    cdf2 = jnp.cumsum(weights2)

    n1 = jnp.searchsorted(samples1, z[:-1], side="right")
    n2 = jnp.searchsorted(samples2, z[:-1], side="right")

    f1 = jnp.where(n1 > 0, cdf1[n1 - 1], 0.0)
    f2 = jnp.where(n2 > 0, cdf2[n2 - 1], 0.0)

    return jnp.sum(jnp.abs(f1 - f2) * dz)


def wasserstein2(samples1, samples2, *, weights1=None, weights2=None):
    """
    Weighted 1D Wasserstein-2 distance between two empirical distributions.

    Parameters
    ----------
    samples1, samples2 : Array
        Samples representing two empirical distributions.
    weights1, weights2 : Array, optional
        Nonnegative probability weights associated with `samples1`
        and `samples2`. If omitted, uniform weights are used.

    Returns
    -------
    Array
        Wasserstein-2 distance.
    """
    samples1 = jnp.asarray(samples1)
    samples2 = jnp.asarray(samples2)

    if weights1 is None:
        weights1 = jnp.ones_like(samples1) / samples1.size
    else:
        weights1 = jnp.asarray(weights1)
        weights1 = weights1 / jnp.sum(weights1)

    if weights2 is None:
        weights2 = jnp.ones_like(samples2) / samples2.size
    else:
        weights2 = jnp.asarray(weights2)
        weights2 = weights2 / jnp.sum(weights2)

    idx1 = jnp.argsort(samples1)
    idx2 = jnp.argsort(samples2)

    samples1 = samples1[idx1]
    samples2 = samples2[idx2]

    weights1 = weights1[idx1]
    weights2 = weights2[idx2]

    cdf1 = jnp.cumsum(weights1)
    cdf2 = jnp.cumsum(weights2)

    u = jnp.sort(jnp.concatenate([cdf1, cdf2]))
    u0 = jnp.concatenate([jnp.array([0.0], dtype=u.dtype), u[:-1]])
    du = u - u0

    i1 = jnp.searchsorted(cdf1, u0, side="right")
    i2 = jnp.searchsorted(cdf2, u0, side="right")

    i1 = jnp.clip(i1, 0, samples1.shape[0] - 1)
    i2 = jnp.clip(i2, 0, samples2.shape[0] - 1)

    q1 = samples1[i1]
    q2 = samples2[i2]

    return jnp.sqrt(jnp.sum((q1 - q2) ** 2 * du))
