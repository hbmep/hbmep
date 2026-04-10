import os
import tomllib
import logging
from operator import attrgetter
from collections import OrderedDict

import arviz as az
import pandas as pd
import numpy as np
from jax import random, numpy as jnp, Array
import numpyro
from numpyro.infer import MCMC
from sklearn.preprocessing import LabelEncoder

import hbmep as mep
from hbmep.util import timing, site, make_pdf

logger = logging.getLogger(__name__)
SEPARATOR = "__"
DATASET_PLOT = "dataset.pdf"
CURVES_PLOT = "curves.pdf"
PREDICTIVE_PLOT = "predictive.pdf"
SAMPLE_SITES = "sample_sites"
REPARAM_SITES = "reparam_sites"
OBS_SITES = "obs_sites"


class BaseModel():
    intensity: str = ""
    features: list[str] = []
    response: list[str] = []
    num_features: int
    num_response: int
    regressors: list[str]
    variables: dict[str, str | list[str]]
    mcmc_params: dict[str, int | float] = {
        "num_chains": 4,
        "num_warmup": 2000,
        "num_samples": 1000,
        "thinning": 1,
    }
    nuts_params: dict[str, int | float] = {
        "target_accept_prob": 0.8,
        "max_tree_depth": (10, 10),
    }
    use_mixture: bool = False
    mep_response: list[str] = []
    mep_window: list[float] = [0, 1]
    mep_size_window: list[float] = [0, 1]
    mep_adjust: float = 1.
    mep_xoffset: list[float] = [1, 1]
    mep_yoffset: list[float] = [1, 1]

    def __init__(
        self,
        *,
        key: random.key = random.key(0),
        toml_path: str | None = None,
        config: dict | None = None
    ):
        self.name: str = "base_model"
        self.key: random.key = key
        self.build_dir: str = ""
        self._response: list[str] = []
        self._num_response: int | None = None
        self.sample_sites: list[str] = []
        self.deterministic_sites: list[str] = []
        self.reparam_sites: list[str] = []
        self.obs_sites: list[str] = []
        self.trace_sites: dict[str, str] = {}

        if toml_path is not None:
            try:
                with open(toml_path, "rb") as f:
                    config = tomllib.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load TOML file: {e}")

        if config is not None:
            self._update_config(config)

    def _update_config(self, config: dict):
        for key, value in config.get("variables", {}).items():
            setattr(self, key, value)
        for key, value in config.get("mcmc", {}).items():
            self.mcmc_params[key] = value
        for key, value in config.get("nuts", {}).items():
            self.nuts_params[key] = value
        for key, value in config.get("mep_data", {}).items():
            setattr(self, key, value)

    def _update_sites(self, model_trace):
        sites = {u: v["type"] for u, v in model_trace.items()}
        sample_sites = [u for u, v in sites.items() if v == "sample" and site.obs not in u.split("_")]
        deterministic_sites = [u for u, v in sites.items() if v == "deterministic" and site.obs not in u.split("_")]
        reparam_sites = [u for u in deterministic_sites if any([u in v.split("_") for v in sample_sites])]
        obs_sites = [u for u in deterministic_sites if u not in sample_sites + reparam_sites]
        obs_sites += [u for u in sites.keys() if site.obs in u.split("_")]
        obs_sites = list(set(obs_sites))
        self.sample_sites = sample_sites
        self.deterministic_sites = deterministic_sites
        self.reparam_sites = reparam_sites
        self.obs_sites = obs_sites
        self.trace_sites = sites

    def _on_response_changed(self, old: list[str] | None, new: list[str]) -> None:
        # Subclass hook: react when response is changed
        ...

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
            if isinstance(response, str):
                response = [response]
            else:
                raise ValueError("Response must be a list of strings")

        if not len(response):
            raise ValueError("Response must have length greater than 0")

        old = self._response
        self._response = response
        self._num_response = None
        self._on_response_changed(old, response)

    @property
    def num_response(self):
        if self._num_response is None:
            self._num_response = len(self.response)

        return self._num_response

    @property
    def mep_data(self):
        attributes = [
            "mep_response",
            "mep_window",
            "mep_size_window",
            "mep_adjust",
            "mep_xoffset",
            "mep_yoffset",
        ]
        return {attr: getattr(self, attr) for attr in attributes}

    @property
    def sites(self):
        attributes = [
            "sample_sites",
            "deterministic_sites",
            "reparam_sites",
            "obs_sites"
        ]
        return {attr: getattr(self, attr) for attr in attributes}

    def get_regressors(self, df: pd.DataFrame):
        return mep.get_regressors(df, **self.variables)

    def get_response(self, df: pd.DataFrame):
        return mep.get_response(df, **self.variables)

    def get_features(self, df: pd.DataFrame):
        return df[self.features].apply(tuple, axis=1)

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
                    df[feature].apply(lambda x: SEPARATOR.join(map(str, x)), axis=1)
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
    def gamma_likelihood(mu, c1, c2):
        def body_gamma_rate(mu, c1, c2):
            z = 1 / (c2 * mu)
            z = (1 / c1) + z
            return z

        def body_gamma_concentration(mu, beta):
            return beta * mu

        beta = body_gamma_rate(mu, c1, c2)
        alpha = body_gamma_concentration(mu, beta)
        return alpha, beta

    @timing
    def trace(
        self,
        df: pd.DataFrame,
        key: Array | None = None,
        **kw
    ):
        trace = mep.trace(
            self.key if key is None else key,
            self._model,
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
        init_params=None,
        key: Array | None = None,
        **kw
    ) -> tuple[MCMC, dict]:
        if not self.sample_sites:
            model_trace = self.trace(df, key=key, **kw)
            self._update_sites(model_trace)
        logger.info(f"Running...")
        mcmc = mep.run(
            self.key if key is None else key,
            self._model,
            *self.get_regressors(df),
            *self.get_response(df),
            nuts_params=self.nuts_params,
            mcmc_params=self.mcmc_params,
            extra_fields=extra_fields,
            init_params=init_params,
            **kw
        )
        posterior = mcmc.get_samples()
        posterior = {k: np.array(v) for k, v in posterior.items()}
        return mcmc, posterior

    @timing
    def make_prediction_dataset(
        self,
        df: pd.DataFrame,
        num_points: int = 100,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
        scale: str = "linear"
    ):
        return mep.make_prediction_dataset(
            df,
            **self.variables,
            num_points=num_points,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            scale=scale
        )

    @timing
    def predict(
        self,
        df: pd.DataFrame,
        posterior: dict | None = None,
        num_samples: int = 100,
        return_sites: list[str] | None = None,
        key: Array | None = None,
        **kw
    ):
        # Generate predictions
        predictive = mep.predict(
            self.key if key is None else key,
            self._model,
            *self.get_regressors(df),
            posterior=posterior,
            num_samples=num_samples,
            return_sites=return_sites,
            **kw
        )
        predictive = {u: np.array(v) for u, v in predictive.items()}
        return predictive

    @timing
    def summary(
        self,
        samples: dict,
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
        var_names = [u for u in var_names if u in samples.keys()]
        samples = {
            u: v.reshape(self.mcmc_params["num_chains"], -1, *v.shape[1:])
            for u, v in samples.items()
        }
        return az.summary(samples, var_names=var_names, hdi_prob=prob, **kwargs)

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
        if not output_path and not self.build_dir:
            logger.info(f"Skipping plotting because output_path not provided.")
            return
        if output_path is None: output_path = os.path.join(self.build_dir, DATASET_PLOT)
        logger.info("Plotting dataset...")
        logger.info(output_path)
        figures = mep.plot(
            df=df,
            **self.variables,
            encoder=encoder,
            mep_array=mep_array,
            **self.mep_data,
            **kw
        )
        figures = [u for u, _ in figures]
        make_pdf(figures=figures, output_path=output_path)
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
        if not output_path and not self.build_dir:
            logger.info(f"Skipping plotting because output_path not provided.")
            return
        if output_path is None: output_path = os.path.join(self.build_dir, CURVES_PLOT)
        logger.info("Plotting curves...")
        logger.info(output_path)
        threshold = (
            posterior[posterior_var]
            if posterior is not None and posterior_var in posterior.keys()
            else None
        )
        figures = mep.plot(
            df=df,
            **self.variables,
            encoder=encoder,
            mep_array=mep_array,
            **self.mep_data,
            prediction_df=prediction_df,
            prediction=predictive[prediction_var],
            prediction_prob=prediction_prob,
            threshold=threshold,
            **kw
        )
        figures = [u for u, _ in figures]
        make_pdf(figures=figures, output_path=output_path)
        return

    @timing
    def plot_predictive(
        self,
        df: pd.DataFrame,
        *,
        prediction_df: pd.DataFrame,
        predictive: dict,
        prediction_var: str = site.obs,
        prediction_prob: float = .95,
        encoder: dict[str, LabelEncoder] | None = None,
        output_path: str | None = None,
        **kw
    ):
        if not output_path and not self.build_dir:
            logger.info(f"Skipping plotting because output_path not provided.")
        if output_path is None: output_path = os.path.join(self.build_dir, PREDICTIVE_PLOT)
        logger.info("Plotting predictive...")
        logger.info(output_path)
        figures = mep.plot(
            df=df,
            **self.variables,
            output_path=output_path,
            encoder=encoder,
            prediction_df=prediction_df,
            prediction=predictive[prediction_var],
            prediction_prob=prediction_prob,
            **kw
        )
        figures = [u for u, _ in figures]
        make_pdf(figures=figures, output_path=output_path)
        return

    def state_dict(self) -> dict:
        key_data = random.key_data(self.key)
        return {
            "name": self.name,
            "variables": self.variables,
            "mcmc_params": self.mcmc_params,
            "nuts_params": self.nuts_params,
            "mep_data": self.mep_data,
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
            "build_dir": self.build_dir,
            "model_name": (
                None if getattr(self, "_model", None) is None
                else self._model.__name__
            ),
        }

    def load_state_dict(self, state: dict):
        self.name = state.get("name", "base_model")
        self._update_config({
            "variables": state.get("variables", {}),
            "mcmc": state.get("mcmc_params", {}),
            "nuts": state.get("nuts_params", {}),
            "mep_data": state.get("mep_data", {}),
        })
        key_data = state.get("key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("key_dtype", jnp.uint32))
        self.key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))
        self.build_dir = state.get("build_dir", self.build_dir)
        model_name = state.get("model_name", None)
        if model_name is not None:
            self._model = getattr(self, model_name)
        return
