import logging

import pandas as pd
import numpy as np
from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.diagnostics import print_summary

from hbmep.config import Config
from hbmep.plotter import Plotter
from hbmep.model.utils import Site as site
from hbmep.utils import timing, floor, ceil
from hbmep.utils.constants import BASE_MODEL, GAMMA_MODEL

logger = logging.getLogger(__name__)


class BaseModel(Plotter):
    NAME = BASE_MODEL

    def __init__(self, config: Config):
        super(BaseModel, self).__init__(config=config)
        self.random_state = 0
        self.rng_key = random.PRNGKey(self.random_state)
        self.mcmc_params = config.MCMC_PARAMS
        logger.info(f"Initialized {self.NAME}")

    def _get_regressors(self, df: pd.DataFrame):
        intensity = df[[self.intensity]].to_numpy()
        features = df[self.features].to_numpy()
        return intensity, features

    def _get_response(self, df: pd.DataFrame):
        response = df[self.response].to_numpy()
        return response,

    def _model(self, model, features, response_obs=None):
        raise NotImplementedError

    @timing
    def run_trace(self, df: pd.DataFrame):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._get_regressors(df=df), *self._get_response(df=df)
            )
        return trace

    @timing
    def run_inference(
        self,
        df: pd.DataFrame,
        sampler: MCMCKernel = None,
        **kwargs
    ) -> tuple[MCMC, dict]:
        # Set up sampler
        if sampler is None: sampler = NUTS(self._model, **kwargs)
        mcmc = MCMC(sampler, **self.mcmc_params)

        # Run MCMC inference
        logger.info(f"Running inference with {self.NAME} ...")
        mcmc.run(self.rng_key, *self._get_regressors(df=df), *self._get_response(df=df))
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return mcmc, posterior_samples

    @timing
    def make_prediction_dataset(
        self,
        df: pd.DataFrame,
        num_points: int = 100,
        min_intensity: float | None = None,
        max_intensity: float | None = None
    ):
        prediction_df = (
            df
            .groupby(by=self.features)
            .agg({self.intensity: ["min", "max"]})
            .copy()
        )
        prediction_df.columns = (
            prediction_df
            .columns
            .map(lambda x: x[1])
        )
        prediction_df = prediction_df.reset_index().copy()

        if min_intensity is not None:
            prediction_df["min"] = min_intensity
        else:
            prediction_df["min"] = (
                prediction_df["min"]
                .apply(lambda x: floor(x, base=self.base))
            )

        if max_intensity is not None:
            prediction_df["max"] = max_intensity
        else:
            prediction_df["max"] = (
                prediction_df["max"]
                .apply(lambda x: (x, ceil(x, base=self.base)))
                .apply(lambda x: x[0] + self.base if x[0] == x[1] else x[1])
            )

        prediction_df[self.intensity] = (
            prediction_df[["min", "max"]]
            .apply(tuple, axis=1)
            .apply(lambda x: (x[0], x[1], min(2000, ceil((x[1] - x[0]) / 5, base=100)),))
            .apply(lambda x: np.linspace(x[0], x[1], num_points))
        )
        prediction_df = prediction_df.explode(column=self.intensity)[self.regressors].copy()
        prediction_df[self.intensity] = prediction_df[self.intensity].astype(float)
        prediction_df = prediction_df.reset_index(drop=True).copy()
        return prediction_df

    @timing
    def predict(
        self,
        df: pd.DataFrame,
        num_samples: int = 100,
        posterior_samples: dict | None = None,
        return_sites: list[str] | None = None,
        rng_key=None
    ):
        if posterior_samples is None:   # Prior predictive
            predictive = Predictive(
                model=self._model,
                num_samples=num_samples,
                return_sites=return_sites
            )
        else:   # Posterior predictive
            predictive = Predictive(
                model=self._model,
                posterior_samples=posterior_samples,
                return_sites=return_sites
            )

        # Generate predictions
        if rng_key is None: rng_key = self.rng_key
        predictions = predictive(rng_key, *self._get_regressors(df=df))
        predictions = {u: np.array(v) for u, v in predictions.items()}
        return predictions

    @staticmethod
    def print_summary(
        samples: dict,
        prob=0.95,
        group_by_chain=False,
        exclude_raw=True,
        exclude_deterministic=True
    ):
        starting_shape_position = 1
        if group_by_chain: starting_shape_position = 2
        a_shape = samples[site.a].shape[starting_shape_position:]

        keys = samples.keys()
        keep_keys = []
        for key in keys:
            if exclude_raw and "_raw" in key: continue
            if exclude_deterministic:
                key_shape = samples[key].shape[starting_shape_position:]
                if not "-".join(map(str, key_shape)) in "-".join(map(str, a_shape)):
                    continue

            keep_keys.append(key)

        samples = {u: v for u, v in samples.items() if u in keep_keys}
        print_summary(
            samples=samples,
            prob=prob,
            group_by_chain=group_by_chain
        )
        return


class GammaModel(BaseModel):
    NAME = GAMMA_MODEL

    def __init__(self, config: Config):
        super(GammaModel, self).__init__(config=config)

    def rate(self, mu, c_1, c_2):
        return (
            jnp.true_divide(1, c_1)
            + jnp.true_divide(1, jnp.multiply(c_2, mu))
        )

    def concentration(self, mu, beta):
        return jnp.multiply(mu, beta)
