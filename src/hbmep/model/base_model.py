import os
import tomllib
import logging
from operator import attrgetter

import arviz as az
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
import numpyro
from numpyro.infer import MCMC
from sklearn.preprocessing import LabelEncoder

import hbmep as mep
from hbmep.util import timing, site

logger = logging.getLogger(__name__)
SEPARATOR = "__"
DATASET_PLOT = "dataset.pdf"
RC_PLOT = "recruitment_curves.pdf"
SAMPLE_SITES = "sample_sites"
REPARAM_SITES = "reparam_sites"
OBS_SITES = "obs_sites"


class BaseModel():
    intensity: str = ""
    features: list[str] = []
    response: list[str] = []
    mcmc_params: dict[str, int | float] = {
        "num_chains": 4,
        "num_warmup": 2000,
        "num_samples": 1000,
        "thinning": 1,
    }
    nuts_params: dict[str, int | float] = {
        "target_accept_prob": 0.8,
        "max_tree_depth": 10,
    }
    mep_response: list[str] = []
    mep_window: list[float] = [0, 1]
    mep_size_window: list[float] = [0, 1]
    mep_adjust: float = 1.

    def __init__(
        self,
        *,
		toml_path: str | None = None,
		config: dict | None = None
    ):
        self.nam: str = "base_model"
        self.build_dir: str = ""
        self.random_state: int = 0
        self.sites: dict[str, list[str]] = {}

        if toml_path is not None:
            try:
                with open(toml_path, "rb") as f:
                    config = tomllib.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load TOML file: {e}")

        if config is not None: self._update_config(config)

    def _update_config(self, config: dict):
        for key, value in config.get("variables", {}).items():
            setattr(self, key, value)
        for key, value in config.get("mcmc", {}).items():
            self.mcmc_params[key] = value
        for key, value in config.get("nuts", {}).items():
            self.nuts_params[key] = value
        for key, value in config.get("mep_metadata", {}).items():
            setattr(self, key, value)
            
    def _update_sites(self, mcmc: MCMC, posterior):
        if not self.sites:
            sample_sites = list(attrgetter(mcmc._sample_field)(mcmc._last_state).keys())
            reparam_sites = [u for u in posterior.keys() if site.raw(u) in sample_sites]
            obs_sites = [u for u in posterior.keys() if u not in sample_sites + reparam_sites]
            self.sites = {
                SAMPLE_SITES: sample_sites,
                REPARAM_SITES: reparam_sites,
                OBS_SITES: obs_sites
            }

    @property
    def key(self):
        return random.key(self.random_state)

    @key.setter # TODO: Check this
    def key(self, random_state):
        if not isinstance(random_state, int):
            raise ValueError("New random state must be an integer")
        self.random_state = random_state

    @property
    def variables(self):
        attributes = ["intensity", "features", "response"]
        return {attr: getattr(self, attr) for attr in attributes}

    @property
    def regressors(self):
        return [self.intensity] + self.features

    @property
    def num_features(self):
        return len(self.features)

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, response):
        isListOfStrings = (
            isinstance(response, list)
            and all(isinstance(r, str) for r in response)
        )

        if not isListOfStrings:
            raise ValueError("Response must be a list of strings")

        if not len(response):
            raise ValueError("Response must have length greater than 0")

        self._response = response
        self._num_response = None

    @property
    def num_response(self):
        if self._num_response is None:
            self._num_response = len(self.response)

        return self._num_response

    @property
    def mep_metadata(self):
        attributes = [
            "mep_response",
            "mep_window",
            "mep_size_window",
            "mep_adjust"
        ]
        return {attr: getattr(self, attr) for attr in attributes}

    @property
    def sample_sites(self):
        return self.sites.get(SAMPLE_SITES, [])

    @property
    def reparam_sites(self):
        return self.sites.get(REPARAM_SITES, [])

    @property
    def obs_sites(self):
        return self.sites.get(OBS_SITES, [])

    def get_regressors(self, df: pd.DataFrame):
        return mep.get_regressors(df, **self.variables)

    def get_response(self, df: pd.DataFrame):
        return mep.get_response(df, **self.variables)

    @timing
    def load(
        self,
		df: pd.DataFrame,
		mask_non_positive: bool = True
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        # Concatenate (necessary) features
        for i, feature in enumerate(self.features):
            if isinstance(feature, list):
                self.features[i] = SEPARATOR.join(feature)
                df[self.features[i]] = (
                    df[feature].apply(lambda x: SEPARATOR.join(x), axis=1)
                )
                logger.info(f"Concatenated {feature} to {self.features[i]}")

        df, encoder = mep.load(
            df,
            **self.variables,
            mask_non_positive=mask_non_positive
        )
        return df, encoder

    def _model(self, intensity, features, response=None, **kw):
        raise NotImplementedError

    @staticmethod
    def gamma_rate(mu, c1, c2):
        z = 1 / (c2 * mu)
        z = (1 / c1) + z
        return z

    @staticmethod
    def gamma_concentration(mu, beta):
        return beta * mu

    def gamma_likelihood(self, fn, x, args, c1, c2):
        mu = fn(x, *args)
        beta = self.gamma_rate(mu, c1, c2)
        alpha = self.gamma_concentration(mu, beta)
        return mu, alpha, beta

    @timing
    def trace(self, df: pd.DataFrame, **kw):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self.get_regressors(df),
                *self.get_response(df),
                **kw
            )
        return trace

    @timing
    def run(
        self,
        df: pd.DataFrame,
        mcmc: MCMC = None,
        extra_fields: list | tuple = (),
        init_params = None,
        **kw
    ) -> tuple[MCMC, dict]:
        mcmc, posterior = mep.run(
            self.key,
            self._model,
            *self.get_regressors(df),
            *self.get_response(df),
            nuts_params=self.nuts_params,
            mcmc_params=self.mcmc_params,
            extra_fields=extra_fields,
            init_params=init_params,
            **kw
        )
        self._update_sites(mcmc, posterior)
        return mcmc, posterior

    @timing
    def make_prediction_dataset(
        self,
        df: pd.DataFrame,
        num_points: int = 100,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
    ):
        return mep.make_prediction_dataset(
            df,
            **self.variables,
            num_points=num_points,
            min_intensity=min_intensity,
            max_intensity=max_intensity
        )

    @timing
    def predict(
        self,
        df: pd.DataFrame,
        posterior: dict | None = None,
        num_samples: int = 100,
        return_sites: list[str] | None = None
    ):
        # Generate predictions
        predictive = mep.predict(
            self.key,
            self._model,
            *self.get_regressors(df),
            posterior=posterior,
            num_samples=num_samples,
            return_sites=return_sites
        )
        predictive = {u: np.array(v) for u, v in predictive.items()}
        return predictive

    @timing
    def summary(
        self,
        posterior: dict,
        *,
        var_names: list[str] | None = None,
        prob=0.95,
        exclude_deterministic=True,
        **kwargs
    ):
        if var_names is None: var_names = (
            self.sample_sites if exclude_deterministic
            else self.sample_sites + self.deterministic_sites
        )
        var_names = [u for u in var_names if u in posterior.keys()]
        posterior = {
            u: v.reshape(self.mcmc_params["num_chains"], -1, *v.shape[1:])
            for u, v in posterior.items()
        }
        return az.summary(posterior, var_names=var_names, hdi_prob=prob, **kwargs)

    def print_summary(
        self,
        samples: dict,
        var_names: list[str] | None = None,
        prob=0.95,
        exclude_deterministic=True,
        **kwargs
    ):
        summary_df = self.summary(
            samples=samples,
            var_names=var_names,
            prob=prob,
            exclude_deterministic=exclude_deterministic,
            **kwargs
        )
        logger.info(f"Summary\n{summary_df.to_string()}")
        return

    @timing
    def plot(
        self,
        df: pd.DataFrame,
        *,
        encoder: dict[str, LabelEncoder] | None = None,
        mep_array: np.ndarray | None = None,
        output_path: str | None = None,
        **kw
    ):
        if output_path is None: output_path = os.path.join(self.build_dir, DATASET_PLOT)
        logger.info("Plotting dataset...")
        mep.plotter(
            df=df,
            **self.variables,
            output_path=output_path,
            encoder=encoder,
            mep_array=mep_array,
            **self.mep_metadata,
            **kw
        )
        return

    @timing
    def plot_curves(
        self,
        df: pd.DataFrame,
        *,
        prediction_df: pd.DataFrame,
        predictive: dict,
        posterior: dict | None = None,
        prediction_var: str = site.mu,
        prediction_prob: float = 0,
        posterior_var: str = site.a,
        encoder: dict[str, LabelEncoder] | None = None,
        mep_array: np.ndarray | None = None,
        output_path: str | None = None,
        **kw        
    ):
        if output_path is None: output_path = os.path.join(self.build_dir, RC_PLOT)
        logger.info("Plotting curves...")
        mep.plotter(
            df=df,
            **self.variables,
            output_path=output_path,
            encoder=encoder,
            mep_array=mep_array,
            **self.mep_metadata,
            prediction_df=prediction_df,
            prediction=predictive[prediction_var],
            prediction_prob=prediction_prob,
            threshold=posterior[posterior_var] if posterior is not None else None,
            **kw
        )
        return
