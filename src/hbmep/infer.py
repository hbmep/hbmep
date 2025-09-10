import logging
from collections.abc import Callable

import pandas as pd
import numpy as np
from jax import random

import numpyro as pyro
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.infer.inspect import get_dependencies as get_deps

logger = logging.getLogger(__name__)


def get_regressors(
    df: pd.DataFrame,
    intensity: str,
    features: list[str],
    **kw
):
    return (
        df[[intensity]].to_numpy(),
        df[features].to_numpy()
    )


def get_response(
    df: pd.DataFrame,
    response: list[str],
    **kw
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


def trace(
    key: random.key,
    model: Callable,
    intensity: np.ndarray,
    features: np.ndarray,
    response: np.ndarray,
    **kw
):
    with pyro.handlers.seed(rng_seed=key):
        trace = pyro.handlers.trace(model).get_trace(
            intensity,
            features,
            response,
            **kw
        )
    return trace


def run(
    key: random.key,
    model: Callable,
    intensity: np.ndarray,
    features: np.ndarray,
    response: np.ndarray,
    nuts_params: dict | None = None,
    mcmc_params: dict | None = None,
    mcmc: MCMC | None = None,
    extra_fields: list | tuple = (),
    init_params=None,
    **kw
) -> tuple[MCMC, dict]:
    if mcmc is None:
        kernel = NUTS(model, **nuts_params)
        mcmc = MCMC(kernel, **mcmc_params)
        msg = f"Running..."
    else:
        assert isinstance(mcmc, MCMC)
        if mcmc.last_state is not None:
            msg = f"Resuming from last state..."
            mcmc.post_warmup_state = mcmc.last_state
            key = mcmc.post_warmup_state.key
        else: msg = f"Running with provided MCMC..."

    # Run
    logger.info(msg)
    mcmc.run(
        key,
        intensity,
        features,
        response,
        extra_fields=extra_fields,
        init_params=init_params,
        **kw
    )
    posterior = mcmc.get_samples()
    posterior = {k: np.array(v) for k, v in posterior.items()}
    return mcmc, posterior


def predict(
    key: random.key,
    model: Callable,
    intensity: np.ndarray,
    features: np.ndarray,
    posterior: dict | None = None,
    num_samples: int = 100,
    return_sites: list[str] | None = None,
    **kw
):
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
    predictive = {u: np.array(v) for u, v in predictive.items()}
    return predictive
