import logging
from collections.abc import Callable

import pandas as pd
import numpy as np
from jax import random, numpy as jnp

import numpyro as pyro
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.infer.inspect import get_dependencies as get_deps

logger = logging.getLogger(__name__)


def get_regressors(
    df: pd.DataFrame,
    *,
    intensity: str,
    features: list[str],
    response: list[str] | None = None,
):
    if not intensity:
        intensity = []
    else:
        intensity = [intensity]
    return (
        df[intensity].to_numpy(),
        df[features].to_numpy()
    )


def get_response(
    df: pd.DataFrame,
    *,
    response: list[str],
    intensity: str | None = None,
    features: list[str] | None = None,
):
    return df[response].to_numpy(),


def get_dependencies(
    model: Callable,
    df,
    intensity: str,
    features: list[str],
    response: list[str],
):
    return get_deps(
        model,
        get_regressors(df, intensity, features),
        get_response(df, response)
    )


def trace(key: random.key, model: Callable, *args, **kw):
    r"""
    Run model trace.

    :param random.key key: Random number generator for sampling.
    :param Callable key: Numpyro model.
    :param args: Arguments passed to the `model`.
    :param kw: Keyword arguments passed to the `model`. 
    """
    with pyro.handlers.seed(rng_seed=key):
        trace = pyro.handlers.trace(model).get_trace(*args, **kw)
    return trace


def run(
    key: random.key,
    model: Callable,
    intensity: np.ndarray | jnp.ndarray,
    features: np.ndarray | jnp.ndarray,
    response: np.ndarray | jnp.ndarray,
    nuts_params: dict | None = None,
    mcmc_params: dict | None = None,
    mcmc: MCMC | None = None,
    extra_fields: list | tuple = (),
    init_params=None,
    **kw
) -> tuple[MCMC, dict]:
    r"""
    Run model.

    :param random.key key: Random number generator for sampling.
    :param Callable model: numpyro model.
    :param intensity: np.ndarray | jnp.ndarray: 
        Intensity array of shape (N, 1), N = number of observations.
    :param features: np.ndarray | jnp.ndarray:
        Features array of shape (N, F), F = number of feature variables.
    :param response: np.ndarray | jnp.ndarray:
        Response array of shape (N, R), R = number of response variables.
    :param nuts_params: dict | None:
        Keyword arguments passed to numpyro.infer.NUTS. Defaults to {}.
    :param mcmc_params: dict | None:
        Keyword arguments passed to numpyro.infer.MCMC. Defaults to {}.
    :param extra_fields: list | tuple
    :param init_params
    :param kw: Keyword arguments passed to the `model`.
    """
    nuts_params = nuts_params or {}
    mcmc_params = mcmc_params or {}
    if mcmc is None:
        kernel = NUTS(model, **nuts_params)
        mcmc = MCMC(kernel, **mcmc_params)
    mcmc.run(
        key,
        intensity,
        features,
        response,
        extra_fields=extra_fields,
        init_params=init_params,
        **kw
    )
    return mcmc


def predict(
    key: random.key,
    model: Callable,
    intensity: np.ndarray | jnp.ndarray,
    features: np.ndarray | jnp.ndarray,
    posterior: dict | None = None,
    num_samples: int = 100,
    return_sites: list[str] | None = None,
    **kw
):
    r"""
    Generate predictive distribution.

    :param random.key key: Random number generator for sampling.
    :param Callable model: numpyro model.
    :param intensity: np.ndarray | jnp.ndarray: 
        Intensity array of shape (N, 1), N = number of observations.
    :param features: np.ndarray | jnp.ndarray:
        Features array of shape (N, F), F = number of feature variables.
    :param posterior: dict | None:
        Dictionary of posterior samples.
        Gf not provided, samples are generated from prior predictive.
    :param extra_fields: list | tuple
    :param init_params
    :param kw: Keyword arguments passed to the `model`.
    """
    if posterior is None:               # prior predictive
        predictive_fn = Predictive(
            model=model,
            num_samples=num_samples,
            return_sites=return_sites
        )
    else:                               # posterior predictive
        predictive_fn = Predictive(
            model=model,
            posterior_samples=posterior,
            return_sites=return_sites
        )
    # Generate predictions
    predictive = predictive_fn(key, intensity, features, **kw)
    return predictive
