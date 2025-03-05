import os
import tomllib
import logging
from operator import attrgetter
from collections import defaultdict

import arviz as az
import pandas as pd
import numpy as np
from jax import random
import jax.numpy as jnp
from sklearn.preprocessing import LabelEncoder

import numpyro
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.diagnostics import print_summary

import hbmep as mep
from hbmep.util import Site as site, timing, floor, ceil

logger = logging.getLogger(__name__)
SEPARATOR = "__"
DATASET_PLOT = "dataset.pdf"
RC_PLOT = "recruitment_curves.pdf"


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

    def __init__(self, toml_path: str | None = None, config: dict | None = None):
        self.random_state: int = 0
        self.name: str = "base_model"
        self.build_dir: str = ""
        # Stochastic and deterministic sites (dynamically set when the model is first run)
        self.sample_sites: list[str] | None = None
        self.deterministic_sites: list[str] | None = None

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

    @property
    def key(self): return random.PRNGKey(self.random_state)

    @key.setter
    def key(self, random_state):
        if not isinstance(random_state, int):
            raise ValueError("New random state must be an integer")
        self.random_state = random_state

    @property
    def variables(self):
        attributes = ["intensity", "features", "response"]
        return {attr: getattr(self, attr) for attr in attributes}

    @property
    def regressors(self): return [self.intensity] + self.features

    @property
    def num_features(self): return len(self.features)

    @property
    def response(self): return self._response

    @response.setter
    def response(self, response):
        isStr = isinstance(response, str)
        isListOfStrings = isinstance(response, list) and all(isinstance(r, str) for r in response)

        if not (isStr or isListOfStrings):
            raise ValueError("Response must be a string, or a list of strings")

        if isListOfStrings and not len(response):
            raise ValueError("Response as a list must have length greater than 0")

        self._response = response
        self._num_response = None

    @property
    def num_response(self):
        if self._num_response is None:
            self._num_response = len(self.response) if isinstance(self.response, list) else 1
        return self._num_response

    @property
    def mep_metadata(self):
        attributes = ["mep_response", "mep_window", "mep_size_window", "mep_adjust"]
        return {attr: getattr(self, attr) for attr in attributes}

    def _get_regressors(self, df: pd.DataFrame):
        intensity = df[self.intensity].to_numpy()
        features = df[self.features].to_numpy()
        return intensity, features

    def _get_response(self, df: pd.DataFrame):
        response = df[self.response].to_numpy()
        return response,

    @timing
    def load(self, df: pd.DataFrame, mask_non_positive: bool = True) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        if self.build_dir:
            os.makedirs(self.build_dir, exist_ok=True)
            logger.info(f"Build directory {self.build_dir}")

        # Concatenate (necessary) features
        for i, feature in enumerate(self.features):
            if isinstance(feature, list):
                self.features[i] = SEPARATOR.join(feature)
                df[self.features[i]] = df[feature].apply(lambda x: SEPARATOR.join(x), axis=1)
                logger.info(f"Concatenated {feature} to {self.features[i]}")

        # Positive response constraint
        if mask_non_positive:
            non_positive_obs = df[self.response].values <= 0
            num_non_positive_obs = non_positive_obs.sum()
            if num_non_positive_obs:
                df[self.response] = np.where(non_positive_obs, np.nan, df[self.response].values)
                logger.info(f"Masked {num_non_positive_obs} non-positive observations")

        df, encoder = mep.fit_transform(df=df, features=self.features)
        return df, encoder

    def _model(self, intensity, features, response_obs=None, **kwargs):
        raise NotImplementedError

    def gamma_rate(self, mu, c1, c2):
        return jnp.true_divide(1, c1) + jnp.true_divide(1, jnp.multiply(c2, mu))

    def gamma_concentration(self, mu, beta):
        return jnp.multiply(mu, beta)

    @timing
    def trace(self, df: pd.DataFrame, **kwargs):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._get_regressors(df=df), *self._get_response(df=df), **kwargs
            )
        return trace

    @timing
    def run(
        self,
        df: pd.DataFrame,
        mcmc: MCMC = None,
        extra_fields: list | tuple = (),
        **kwargs
    ) -> tuple[MCMC, dict]:
        key = self.key
        if mcmc is None:
            msg = f"Running {self.name}..."
            kernel = NUTS(self._model, **self.nuts_params)
            mcmc = MCMC(kernel, **self.mcmc_params)
        else:
            assert isinstance(mcmc, MCMC)
            if mcmc.last_state is not None:
                msg = f"Resuming {self.name} from last state..."
                mcmc.post_warmup_state = mcmc.last_state
                key = mcmc.post_warmup_state.key
            else:
                msg = f"Running {self.name} with provided MCMC..."

        # Run MCMC
        logger.info(msg)
        mcmc.run(
            key,
            *self._get_regressors(df=df),
            *self._get_response(df=df),
            extra_fields=extra_fields,
            **kwargs
        )
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        sample_sites = list(attrgetter(mcmc._sample_field)(mcmc._last_state).keys())
        sample_sites_shapes = [posterior_samples[u].shape for u in sample_sites]
        deterministic_sites = [
            u for u in posterior_samples.keys()
            if (u not in sample_sites) and (posterior_samples[u].shape in sample_sites_shapes)
        ]
        if self.sample_sites is None: self.sample_sites = sample_sites
        if self.deterministic_sites is None: self.deterministic_sites = deterministic_sites
        return mcmc, posterior_samples

    @timing
    def make_prediction_dataset(self, df: pd.DataFrame, *args, **kw):
        return mep.make_prediction_dataset(
            df, intensity=self.intensity, features=self.features, *args, *kw
        )

    @timing
    def predict(
        self,
        df: pd.DataFrame,
        *,
        num_samples: int = 100,
        posterior: dict | None = None,
        return_sites: list[str] | None = None,
        key=None
    ):
        if posterior is None:   # Prior predictive
            predictive_fn = Predictive(
                model=self._model,
                num_samples=num_samples,
                return_sites=return_sites
            )
        else:   # Posterior predictive
            predictive_fn = Predictive(
                model=self._model,
                posterior_samples=posterior,
                return_sites=return_sites
            )

        # Generate predictions
        if key is None: key = self.key
        predictive = predictive_fn(key, *self._get_regressors(df=df))
        predictive = {u: np.array(v) for u, v in predictive.items()}
        return predictive

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
        *,
        df: pd.DataFrame,
        encoder: dict[str, LabelEncoder] | None = None,
        mep_matrix: np.ndarray | None = None,
        output_path: str | None = None,
        **kwargs
    ):
        if output_path is None: output_path = os.path.join(self.build_dir, DATASET_PLOT)
        if mep_matrix is not None and self.mep_response != self.response:
            idx = [r for r, response in enumerate(self.mep_response) if response in self.response]
            mep_matrix = mep_matrix[..., idx]

        logger.info("Plotting dataset...")
        mep.plot(
            df=df,
            **self.variables,
            output_path=output_path,
            encoder=encoder,
            mep_matrix=mep_matrix,
            **self.mep_metadata,
            **kwargs
        )
        return

    @timing
    def plot_curves(
        self,
        *,
        df: pd.DataFrame,
        posterior: dict,
        prediction_df: pd.DataFrame,
        predictive: dict,
        encoder: dict[str, LabelEncoder] | None = None,
        mep_matrix: np.ndarray | None = None,
        output_path: str | None = None,
        **kwargs        
    ):
        if output_path is None: output_path = os.path.join(self.build_dir, RC_PLOT)
        if mep_matrix is not None and self.mep_response != self.response:
            idx = [r for r, response in enumerate(self.mep_response) if response in self.response]
            mep_matrix = mep_matrix[..., idx]

        threshold = posterior[site.a]
        response_pred = predictive[site.mu]
        logger.info("Plotting recruitment curves...")
        mep.plot(
            df=df,
            **self.variables,
            output_path=output_path,
            encoder=encoder,
            mep_matrix=mep_matrix,
            **self.mep_metadata,
            prediction_df=prediction_df,
            response_pred=response_pred,
            threshold=threshold,
            **kwargs
        )
        return
