import os
import tomllib
import logging
from operator import attrgetter
from collections import defaultdict
# from abc import ABC, abstractmethod

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
REC_PLOT = "recruitment_curves.pdf"


class BaseModel(object):
    def __init__(self, toml_path: str | None = None, config: dict | None = None):
        """
        Initialize the model.

        Parameters
        ----------
        toml_path : str, optional
            The path to the toml file with the model configuration.

        Attributes
        ----------
        random_state : int
            The random state for the model.
        intensity : str
            The name of the intensity variable.
        features : list[str]
            The names of the feature variables.
        response : list[str]
            The names of the response variables.
        sample_sites : list[str], optional
            The names of the model sample sites.
        deterministic_sites : list[str], optional
            The names of the model deterministic sites.
        mcmc : dict[str, int]
            The parameters for the MCMC.
        nuts : dict[str, int]
            The parameters for the NUTS kernel.

        Notes
        -----
        If toml_path is provided, the model will be configured with the
        variables and options defined in the toml file. If not, the model
        will be configured with default values.
        """
        # Model name
        self.name: str = "base_model"
        # Random seed
        self.random_state: int = 0
        # Paths
        self.csv_path: str | None = None
        self.build_dir: str | None = None
        # Variables
        self.intensity: str = ""
        self.features: list[str] = []
        self.response: list[str] = []
        # MCMC and NUTS sampler settings
        self.mcmc_params: dict[str, int] = {}
        self.nuts_params: dict[str, int] = {}
        # Stochastic and deterministic sites will be set when model runs
        self.sample_sites: list[str] | None = None
        self.deterministic_sites: list[str] | None = None
        # Optional MEP data
        self.mep_matrix_path: str | None = None
        self.mep_response: list[str] | None = None
        self.mep_window: list[float] | None = None
        self.mep_size_window: list[float] | None = None
        # Other optional
        self.base = 10

        if toml_path is not None:
            with open(toml_path, "rb") as f: config = tomllib.load(f)

        if config is not None:
            self._init_with_config(config)

        default_mcmc = {
            "num_chains": 4,
            "num_warmup": 2000,
            "num_samples": 1000,
            "thinning": 1,
        }
        default_nuts = {
            "target_accept_prob": 0.8,
            "max_tree_depth": 10,
        }

        for key, value in default_mcmc.items():
            if key not in self.mcmc_params:
                self.mcmc_params[key] = value

        for key, value in default_nuts.items():
            if key not in self.nuts_params:
                self.nuts_params[key] = value

    def _init_with_config(self, config: dict):
        paths = config.get("paths", {})
        variables = config.get("variables", {})
        assert {"intensity", "features", "response"} <= set(variables.keys())
        mcmc_params = config.get("mcmc", {})
        nuts_params = config.get("nuts", {})
        optional = config.get("optional", {})

        for key, value in paths.items(): setattr(self, key, value)
        for key, value in variables.items(): setattr(self, key, value)
        for key, value in optional.items(): setattr(self, key, value)
        self.mcmc_params = mcmc_params
        self.nuts_params= nuts_params
        return

    @property
    def rng_key(self):
        """
        A PRNG key for use with Numpyro. This key is initialized with self.random_state.

        Returns:
            PRNGKey: A PRNG key.
        """
        return random.PRNGKey(self.random_state)

    @property
    def n_response(self):
        """
        The number of response variables in the model.

        Returns:
            int: The number of response variables.
        """
        return len(self.response)

    @property
    def n_features(self):
        """
        The number of feature variables in the model.

        Returns:
            int: The number of feature variables.
        """
        return len(self.features)

    @property
    def variables(self):
        return {
            "intensity": self.intensity,
            "features": self.features,
            "response": self.response
        }

    @property
    def regressors(self):
        return [self.intensity] + self.features

    @property
    def mep_metadata(self):
        return {
            "mep_response": self.mep_response,
            "mep_window": self.mep_window,
            "mep_size_window": self.mep_size_window,
        }

    # @staticmethod
    # def _get_combinations(
    #     df: pd.DataFrame,
    #     columns: list[str],
    #     sort_key=None
    # ) -> list[tuple[int]]:
    #     combinations = (
    #         df[columns]
    #         .apply(tuple, axis=1)
    #         .unique()
    #         .tolist()
    #     )
    #     combinations = sorted(combinations, key=sort_key)
    #     return combinations
    #
    @staticmethod
    def _get_combination_inverse(
        combination: tuple[int],
        columns: list[str],
        encoder_dict: dict[str, LabelEncoder]
    ) -> tuple:
        return tuple(
            encoder_dict[column].inverse_transform(np.array([value]))[0]
            for (column, value) in zip(columns, combination)
        )

    def _get_regressors(self, df: pd.DataFrame):
        intensity = df[[self.intensity]].to_numpy()
        features = df[self.features].to_numpy()
        return intensity, features

    def _get_response(self, df: pd.DataFrame):
        response = df[self.response].to_numpy()
        return response,

    @staticmethod
    def _preprocess(
        df: pd.DataFrame,
        columns: list[str]
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        # Encode
        encoder_dict = defaultdict(LabelEncoder)
        df[columns] = (
            df[columns]
            .apply(lambda x: encoder_dict[x.name].fit_transform(x))
        )
        return df, encoder_dict

    @timing
    def load(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        if self.build_dir:
            os.makedirs(self.build_dir, exist_ok=True)
            logger.info(f"Model output will be saved here - {self.build_dir}")

        # Concatenate (necessary) features
        for i, feature in enumerate(self.features):
            if isinstance(feature, list):
                df[self.features[i]] = (
                    df[feature]
                    .apply(lambda x: SEPARATOR.join(x), axis=1)
                )
                self.features[i] = SEPARATOR.join(feature)
                logger.info(f"Concatenated {feature} to {self.features[i]}")

        # Positive response constraint
        num_non_positive_observation = (
            (df[self.response] <= 0)
            .any(axis=1)
            .sum()
        )
        if num_non_positive_observation:
            logger.info(
                "Total non-positive observations: ",
                f"{num_non_positive_observation}"
            )
        assert not num_non_positive_observation
        df, encoder_dict = self._preprocess(df=df, columns=self.features)
        return df, encoder_dict

    def _model(self, intensity, features, response_obs=None, **kwargs):
        raise NotImplementedError

    def rate(self, mu, c_1, c_2):
        return (
            jnp.true_divide(1, c_1)
            + jnp.true_divide(1, jnp.multiply(c_2, mu))
        )

    def concentration(self, mu, beta):
        return jnp.multiply(mu, beta)

    @timing
    def trace(self, df: pd.DataFrame, **kwargs):
        """
        Run the model and return the trace of the MCMC.

        Parameters:
        df : pd.DataFrame
            The data to use for the MCMC.

        Returns:
        trace : dict
            The trace of the MCMC.
        """
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
        rng_key = self.rng_key
        if mcmc is None:
            msg = f"Running {self.name} ..."
            kernel = NUTS(self._model, **self.nuts_params)
            mcmc = MCMC(kernel, **self.mcmc_params)
        else:
            assert isinstance(mcmc, MCMC)
            if mcmc.last_state is not None:
                msg = f"Resuming {self.name} from last state ..."
                mcmc.post_warmup_state = mcmc.last_state
                rng_key = mcmc.post_warmup_state.rng_key
            else:
                msg = f"Running {self.name} with provided MCMC ..."

        # Run MCMC
        logger.info(msg)
        mcmc.run(
            rng_key,
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
    def make_prediction_dataset(
        self,
        df: pd.DataFrame,
        *,
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
        *,
        num_samples: int = 100,
        posterior: dict | None = None,
        return_sites: list[str] | None = None,
        rng_key=None
    ):
        if posterior is None:   # Prior predictive
            predictive = Predictive(
                model=self._model,
                num_samples=num_samples,
                return_sites=return_sites
            )
        else:   # Posterior predictive
            predictive = Predictive(
                model=self._model,
                posterior_samples=posterior,
                return_sites=return_sites
            )

        # Generate predictions
        if rng_key is None: rng_key = self.rng_key
        predictions = predictive(rng_key, *self._get_regressors(df=df))
        predictions = {u: np.array(v) for u, v in predictions.items()}
        return predictions

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
        mep_matrix: np.ndarray | None = None,
        encoder_dict: dict | None = None,
        output_path: str | None = None,
        **kwargs
    ):
        if output_path is None: output_path = os.path.join(self.build_dir, DATASET_PLOT)
        if mep_matrix is not None and self.mep_response != self.response:
            idx = [r for r, response in enumerate(self.mep_response) if response in self.response]
            mep_matrix = mep_matrix[..., idx]

        logger.info("Plotting dataset ...")
        return mep.plot(
            df,
            **self.variables,
            mep_matrix=mep_matrix,
            **self.mep_metadata,
            encoder_dict=encoder_dict,
            output_path=output_path,
            base=self.base,
            **kwargs
        )

    @timing
    def curveplot(
        self,
        df: pd.DataFrame,
        *,
        posterior: dict,
        df_pred: pd.DataFrame,
        predictive: dict,
        mep_matrix: np.ndarray | None = None,
        encoder_dict: dict | None = None,
        output_path: str | None = None,
        **kwargs
    ):
        if output_path is None: output_path = os.path.join(self.build_dir, REC_PLOT)
        if mep_matrix is not None and mep_matrix.shape[-1] != self.n_response:
            idx = [r for r, response in enumerate(self.mep_response) if response in self.response]
            mep_matrix = mep_matrix[..., idx]

        threshold = posterior[site.a]
        response_pred = predictive[site.mu]
        logger.info("Plotting recruitment curves ...")
        return mep.plot(
            df,
            **self.variables,
            mep_matrix=mep_matrix,
            **self.mep_metadata,
            df_pred=df_pred,
            response_pred=response_pred,
            threshold=threshold,
            encoder_dict=encoder_dict,
            output_path=output_path,
            base=self.base,
            **kwargs
        )

