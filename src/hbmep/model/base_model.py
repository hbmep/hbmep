import os
import tomllib
import logging

import arviz as az
import pandas as pd
import numpy as np
from jax import random, numpy as jnp, Array
from numpyro.infer import MCMC
from sklearn.preprocessing import LabelEncoder

from hbmep.dataset import (
    process as _process,
    make_prediction_dataset as _make_prediction_dataset
)
from hbmep.plotter import plot as _plot
from hbmep.infer import (
    trace as _trace,
    get_regressors as _get_regressors,
    get_response as _get_response,
    run as _run,
    predict as _predict,
)
from hbmep.util import site, timing, make_pdf

logger = logging.getLogger(__name__)

SEPARATOR = "__"
DATASET_PLOT = "dataset.pdf"
CURVES_PLOT = "curves.pdf"


class BaseModel:
    """
    Base class for hbMEP models.

    This class provides core functionality for loading and processing datasets,
    running inference (MCMC) to estimate curves, generating predictions,
    and plotting datasets and estimated curves.

    Notes
    -----
    - Subclasses must implement the `_model` method, which defines
      the probabilistic model.
    """
    def __init__(
        self,
        *,
        key: Array | None = None,
        toml_path: str | None = None,
        config: dict | None = None
    ):
        self.key = random.key(0) if key is None else key
        self.name: str = "base_model"
        self.build_dir: str = ""
        self.use_mixture: bool = False

        self.intensity: str = ""
        self.features: list[str] = []
        self._response: list[str] = []
        self._num_response: int | None = None

        self.mcmc_kw: dict[str, int | float] = {
            "num_chains": 4,
            "num_warmup": 2000,
            "num_samples": 1000,
            "thinning": 1,
        }
        self.nuts_kw: dict[str, int | float] = {
            "target_accept_prob": 0.8,
            "max_tree_depth": (10, 10),
        }

        self.mep_response: list[str] = []
        self.mep_window: list[float] = [0, 1]
        self.mep_size_window: list[float] = [0, 1]
        self.mep_adjust: float = 1.0
        self.mep_xoffset: list[float] = [1, 1]
        self.mep_yoffset: list[float] = [1, 1]

        self.trace_sites: dict[str, str] = {}
        self._sample_sites: list[str] = []
        self._deterministic_sites: list[str] = []
        self._obs_sites: list[str] = []

        if toml_path is not None:
            try:
                with open(toml_path, "rb") as f:
                    config = tomllib.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load TOML file: {e}")

        if config is not None:
            self._update_config(config)

        return

    def _update_config(self, config: dict):
        for key, value in config.get("variables", {}).items():
            setattr(self, key, value)
        for key, value in config.get("mcmc_kw", {}).items():
            self.mcmc_kw[key] = value
        for key, value in config.get("nuts_kw", {}).items():
            self.nuts_kw[key] = value
        for key, value in config.get("mep_data", {}).items():
            setattr(self, key, value)
        return

    def _update_sites(self, model_trace):
        site_types = {name: node["type"] for name, node in model_trace.items()}
        self.trace_sites = site_types
        self._sample_sites = [
            name for name, typ in site_types.items() if (
                typ == "sample"
                and site.obs not in name.split("_")
                and name not in {site.outlier_prob}
            )
        ]
        self._deterministic_sites = [
            name for name, typ in site_types.items()
            if typ == "deterministic"
        ]
        self._obs_sites = [
            name for name in site_types.keys()
            if site.obs in name.split("_")
        ]
        return

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
        return

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
    def sample_sites(self):
        return self._sample_sites

    @property
    def deterministic_sites(self):
        return self._deterministic_sites

    @property
    def obs_sites(self):
        return self._obs_sites

    @property
    def sites(self):
        attributes = [
            "sample_sites",
            "deterministic_sites",
            "obs_sites"
        ]
        return {attr: getattr(self, attr) for attr in attributes}

    def get_regressors(self, df: pd.DataFrame):
        return _get_regressors(df, **self.variables)

    def get_response(self, df: pd.DataFrame):
        return _get_response(df, **self.variables)

    def get_features(self, df: pd.DataFrame):
        return df[self.features].apply(tuple, axis=1)

    @timing
    def process(
        self,
        df: pd.DataFrame,
        mask_non_positive: bool = True
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        # Concatenate (necessary) features
        for i, feature in enumerate(self.features):
            if isinstance(feature, list):
                self.features[i] = SEPARATOR.join(feature)
                df[self.features[i]] = (
                    df[feature].apply(
                        lambda x: SEPARATOR.join(map(str, x)), axis=1
                    )
                )
                logger.info(f"Concatenated {feature} to {self.features[i]}")
        df, encoder = _process(
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
        trace = _trace(
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
        if not self._sample_sites:
            model_trace = self.trace(df, key=key, **kw)
            self._update_sites(model_trace)
        logger.info(f"Running...")
        mcmc = _run(
            self.key if key is None else key,
            self._model,
            *self.get_regressors(df),
            *self.get_response(df),
            nuts_kw=self.nuts_kw,
            mcmc_kw=self.mcmc_kw,
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
        *,
        num_points: int = 100,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
        scale: str = "linear"
    ):
        return _make_prediction_dataset(
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
        *,
        posterior: dict | None = None,
        num_samples: int = 100,
        return_sites: list[str] | None = None,
        key: Array | None = None,
        **kw
    ):
        predictive = _predict(
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
        if var_names is None:
            var_names = (
                self.sample_sites
                if exclude_deterministic
                else self.sample_sites + self.deterministic_sites
            )
        var_names = [u for u in var_names if u in samples.keys()]
        num_chains = self.mcmc_kw["num_chains"]
        reshaped = {
            u: v.reshape(num_chains, -1, *v.shape[1:])
            for u, v in samples.items()
        }
        return az.summary(reshaped, var_names=var_names, hdi_prob=prob, **kwargs)

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
        mep_array: np.ndarray | None = None,
        output_path: str | None = None,
        **kw
    ):
        if not output_path and not self.build_dir:
            logger.info("Skipping plotting because output_path not provided.")
            return

        if output_path is None:
            output_path = os.path.join(self.build_dir, DATASET_PLOT)

        logger.info("Plotting dataset...")
        logger.info(output_path)
        figures = _plot(
            df=df,
            **self.variables,
            encoder=encoder,
            mep_array=mep_array,
            **self.mep_data,
            **kw,
        )
        figures = [fig for fig, _ in figures]
        make_pdf(figures=figures, output_path=output_path)
        return

    @timing
    def plot_curves(
        self,
        *,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        predictive: dict,
        posterior: dict | None = None,
        predictive_var: str = site.mu,
        predictive_hdi_var: str | None = site.obs,
        predictive_hdi_prob: float = 0.0,
        threshold_var: str | None = site.a,
        threshold_hdi_prob: float = 0.95,
        encoder: dict[str, LabelEncoder] | None = None,
        mep_array: np.ndarray | None = None,
        output_path: str | None = None,
        **kw
    ):
        if not output_path and not self.build_dir:
            logger.info("Skipping plotting because output_path not provided.")
            return

        if output_path is None:
            output_path = os.path.join(self.build_dir, CURVES_PLOT)

        logger.info("Plotting curves...")
        logger.info(output_path)
        figures = _plot(
            df=df,
            **self.variables,
            encoder=encoder,
            mep_array=mep_array,
            **self.mep_data,
            prediction_df=prediction_df,
            predictive=predictive,
            posterior=posterior,
            predictive_var=predictive_var,
            predictive_hdi_var=predictive_hdi_var,
            predictive_hdi_prob=predictive_hdi_prob,
            threshold_var=threshold_var,
            threshold_hdi_prob=threshold_hdi_prob,
            **kw,
        )
        figures = [fig for fig, _ in figures]
        make_pdf(figures=figures, output_path=output_path)
        return

    def state_dict(self) -> dict:
        key_data = random.key_data(self.key)
        return {
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
            "name": self.name,
            "build_dir": self.build_dir,
            "use_mixture": self.use_mixture,
            "variables": self.variables,
            "mcmc_kw": self.mcmc_kw,
            "nuts_kw": self.nuts_kw,
            "mep_data": self.mep_data,
            "model_name": (
                None if getattr(self, "_model", None) is None
                else self._model.__name__
            ),
        }

    def load_state_dict(self, state: dict):
        key_data = state.get("key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("key_dtype", jnp.uint32))
        self.key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))
        self.name = state.get("name", "base_model")
        self.build_dir = state.get("build_dir", self.build_dir)
        self.use_mixture = state.get("use_mixture", self.use_mixture)
        self._update_config({
            "variables": state.get("variables", {}),
            "mcmc_kw": state.get("mcmc_kw", {}),
            "nuts_kw": state.get("nuts_kw", {}),
            "mep_data": state.get("mep_data", {}),
        })
        model_name = state.get("model_name", None)
        if model_name is not None:
            if not hasattr(self, model_name):
                raise AttributeError(
                    f"{type(self).__name__} has no "
                    f"method '{model_name}' needed to restore _model."
                )
            self._model = getattr(self, model_name)
        return self
