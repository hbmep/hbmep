import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.diagnostics import hpdi

from hbmep.config import Config
from hbmep.dataset import Dataset
from hbmep.model.utils import Site as site
from hbmep.utils import (
    timing,
    floor,
    ceil
)
from hbmep.utils.constants import (
    BASE_MODEL,
    DATASET_PLOT,
    RECRUITMENT_CURVES,
    PRIOR_PREDICTIVE,
    POSTERIOR_PREDICTIVE,
    MCMC_NC,
    DIAGNOSTICS_CSV,
    LOO_CSV,
    WAIC_CSV
)

logger = logging.getLogger(__name__)


class BaseModel(Dataset):
    NAME = BASE_MODEL

    def __init__(self, config: Config):
        super(BaseModel, self).__init__(config=config)
        self.random_state = 0
        self.rng_key = jax.random.PRNGKey(self.random_state)
        self.mcmc_params = config.MCMC_PARAMS

        self.response_colors = plt.cm.rainbow(np.linspace(0, 1, self.n_response))
        self.base = config.BASE
        self.subplot_cell_width = 5
        self.subplot_cell_height = 3
        self.recruitment_curve_props = {
            "label": "Recruitment Curve", "color": "black", "alpha": 0.4
        }
        self.threshold_posterior_props = {"color": "green", "alpha": 0.4}
        logger.info(f"Initialized {self.NAME}")

    def _model(self, subject, features, intensity, response_obs=None):
        raise NotImplementedError

    def _collect_regressor(self, df: pd.DataFrame):
        subject = df[self.subject].to_numpy().reshape(-1,)
        n_subject = df[self.subject].nunique()

        features = df[self.features].to_numpy().T
        n_features = df[self.features].nunique().tolist()

        intensity = df[self.intensity].to_numpy().reshape(-1,)
        n_data = intensity.shape[0]

        return (subject, n_subject), (features, n_features), (intensity, n_data),

    def _collect_response(self, df: pd.DataFrame):
        response = df[self.response].to_numpy()
        return response,

    def _make_index_from_combination(self, combination: tuple[int]):
        ind = [slice(None)] + list(combination) + [slice(None)]
        return tuple(ind)

    def _collect_samples_at_combination(self, combination: tuple[int], samples: np.ndarray):
        return samples[*self._make_index_from_combination(combination=combination)]

    def mep_renderer(
        self,
        df: pd.DataFrame,
        destination_path: str,
        posterior_samples: dict | None = None,
        prediction_df: pd.DataFrame | None = None,
        posterior_predictive: dict | None = None,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        mep_matrix: np.ndarray | None = None,
        **kwargs
    ):
        """
        **kwargs:
            combination_columns: list[str]
            intensity: str
            response: list[str]
            response_colors: list[str] | np.ndarray
            base: int
            subplot_cell_width: int
            subplot_cell_height: int
            recruitment_curve_props: dict
            threshold_posterior_props: dict
        """
        combination_columns = kwargs.get("combination_columns", self.combination_columns)
        intensity = kwargs.get("intensity", self.intensity)
        response = kwargs.get("response", self.response)
        response_colors = kwargs.get("response_colors", self.response_colors)

        base = kwargs.get("base", self.base)
        subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width)
        subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height)
        recruitment_curve_props = kwargs.get("recruitment_curve_props", self.recruitment_curve_props)
        threshold_posterior_props = kwargs.get("threshold_posterior_props", self.threshold_posterior_props)

        if mep_matrix is not None:
            assert mep_matrix.shape[0] == df.shape[0]
            a, b = self.mep_window
            time = np.linspace(a, b, mep_matrix.shape[1])
            within_mep_size_window = (time > self.mep_size_window[0]) & (time < self.mep_size_window[1])

        if posterior_samples is not None:
            assert (prediction_df is not None) and (posterior_predictive is not None)
            mu_posterior_predictive = posterior_predictive[site.mu]

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=combination_columns)
        n_combinations = len(combinations)
        n_response = len(response)

        n_columns_per_response = 1
        if mep_matrix is not None: n_columns_per_response += 1
        if posterior_samples is not None: n_columns_per_response += 2

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Iterate over pdf pages """
        pdf = PdfPages(destination_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )
            fig, axes = plt.subplots(
                nrows=n_rows_current_page,
                ncols=n_fig_columns,
                figsize=(
                    n_fig_columns * subplot_cell_width,
                    n_rows_current_page * subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._invert_combination(
                        combination=curr_combination,
                        columns=combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                """ Filter dataframe based on current combination """
                df_ind = df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                if posterior_samples is not None:
                    """ Filter prediction dataframe based on current combination """
                    prediction_df_ind = prediction_df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                    curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                    """ Predictions for current combination """
                    curr_mu_posterior_predictive = mu_posterior_predictive[:, prediction_df_ind, :]
                    curr_mu_posterior_predictive_map = curr_mu_posterior_predictive.mean(axis=0)

                    """ Threshold estimate for current combination """
                    curr_threshold_posterior = self._collect_samples_at_combination(
                        combination=curr_combination, samples=posterior_samples[site.a]
                    )
                    curr_threshold_map = curr_threshold_posterior.mean(axis=0)
                    curr_threshold_hpdi = hpdi(curr_threshold_posterior, prob=0.95)

                """ Tickmarks """
                min_intensity, max_intensity_ = curr_df[intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=base)
                max_intensity = ceil(max_intensity_, base=base)
                if max_intensity == max_intensity_:
                    max_intensity += base
                curr_x_ticks = np.arange(min_intensity, max_intensity, base)

                axes[i, 0].set_xlabel(intensity)
                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (base // 2), right=max_intensity + (base // 2))

                """ Iterate over responses """
                j = 0
                for r, response_muscle in enumerate(response):
                    """ Labels """
                    prefix = f"{tuple(list(curr_combination)[::-1] + [r])}: {response_muscle} - MEP"
                    if not j: prefix = curr_combination_inverse + prefix

                    """ MEP data """
                    if mep_matrix is not None:
                        postfix = " - MEP"
                        ax = axes[i, j]
                        mep_response_ind = [i for i, _response_muscle in enumerate(self.mep_response) if _response_muscle == response_muscle][0]
                        curr_mep_matrix = mep_matrix[df_ind, :, mep_response_ind]
                        max_amplitude = curr_mep_matrix[..., within_mep_size_window].max()

                        for k in range(curr_mep_matrix.shape[0]):
                            x = (curr_mep_matrix[k, :] / max_amplitude) * (base // 2)
                            x += curr_df[intensity].values[k]
                            ax.plot(x, time, color=response_colors[r], alpha=.4)

                        ax.axhline(
                            y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
                        )
                        ax.axhline(
                            y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
                        )
                        ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + (self.mep_size_window[0] - (-0.001)))

                        ax.set_ylabel("Time")
                        ax.set_title(prefix + postfix)
                        ax.sharex(axes[i, 0])
                        ax.tick_params(axis="x", rotation=90)
                        if j > 0 and ax.get_legend() is not None: ax.get_legend().remove()
                        j += 1

                    """ MEP Size scatter plot """
                    postfix = " - MEP Size"
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)

                    ax.set_ylabel(response_muscle)
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    if posterior_samples is not None:
                        """ MEP Size scatter plot and recruitment curve """
                        postfix = "Recruitment Curve Fit"
                        ax = axes[i, j]
                        sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
                        sns.lineplot(
                            x=curr_prediction_df[intensity],
                            y=curr_mu_posterior_predictive_map[:, r],
                            ax=ax,
                            **recruitment_curve_props,
                        )
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **threshold_posterior_props
                        )

                        ax.set_title(postfix)
                        ax.sharex(axes[i, 0])
                        ax.sharey(axes[i, j - 1])
                        ax.tick_params(axis="x", rotation=90)
                        j += 1

                        """ Threshold KDE """
                        ax = axes[i, j]
                        postfix = "Threshold Estimate"
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **threshold_posterior_props
                        )
                        ax.axvline(
                            curr_threshold_map[r],
                            linestyle="--",
                            color=response_colors[r],
                            label="Threshold"
                        )
                        ax.axvline(
                            curr_threshold_hpdi[:, r][0],
                            linestyle="--",
                            color="black",
                            alpha=.4,
                            label="95% HPDI"
                        )
                        ax.axvline(
                            curr_threshold_hpdi[:, r][1],
                            linestyle="--",
                            color="black",
                            alpha=.4
                        )

                        ax.set_xlabel(intensity)
                        ax.set_title(postfix)
                        if j > 0 and ax.get_legend(): ax.get_legend().remove()
                        j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return

    @timing
    def plot(
        self,
        df: pd.DataFrame,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        destination_path: str | None = None,
        **kwargs
    ):
        if destination_path is None: destination_path = os.path.join(
            self.build_dir, DATASET_PLOT
        )
        logger.info("Rendering ...")
        return self.mep_renderer(
            df=df,
            destination_path=destination_path,
            encoder_dict=encoder_dict,
            **kwargs
        )

    @timing
    def render_recruitment_curves(
        self,
        df: pd.DataFrame,
        posterior_samples: dict,
        prediction_df: pd.DataFrame,
        posterior_predictive: dict,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        destination_path: str | None = None,
        **kwargs
    ):
        if destination_path is None: destination_path = os.path.join(
            self.build_dir, RECRUITMENT_CURVES
        )
        logger.info("Rendering recruitment curves ...")
        return self.mep_renderer(
            df=df,
            destination_path=destination_path,
            posterior_samples=posterior_samples,
            prediction_df=prediction_df,
            posterior_predictive=posterior_predictive,
            encoder_dict=encoder_dict,
            **kwargs
        )

    @timing
    def predictive_checks_renderer(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        predictive: dict,
        destination_path: str,
        encoder_dict: dict[str, LabelEncoder] | None = None
    ):
        """ Prior / Posterior predictive samples """
        obs, mu = predictive[site.obs], predictive[site.mu]

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.combination_columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Iterate over pdf pages """
        pdf = PdfPages(destination_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )
            fig, axes = plt.subplots(
                nrows=n_rows_current_page,
                ncols=n_fig_columns,
                figsize=(
                    n_fig_columns * self.subplot_cell_width,
                    n_rows_current_page * self.subplot_cell_height
                ),
                sharex="row",
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._invert_combination(
                        combination=curr_combination,
                        columns=self.combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                """ Filter dataframe based on current combination """
                df_ind = df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                """ Filter prediction dataframe based on current combination """
                prediction_df_ind = prediction_df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                """ Predictive for current combination """
                curr_obs = obs[:, prediction_df_ind, :]
                curr_obs_map = curr_obs.mean(axis=0)

                curr_mu = mu[:, prediction_df_ind, :]
                curr_mu_map = curr_mu.mean(axis=0)

                """ HPDI intervals """
                curr_obs_hpdi_95 = hpdi(curr_obs, prob=.95)
                curr_obs_hpdi_85 = hpdi(curr_obs, prob=.85)
                curr_obs_hpdi_65 = hpdi(curr_obs, prob=.65)

                curr_mu_hpdi_95 = hpdi(curr_mu, prob=.95)

                """ Tickmarks """
                min_intensity, max_intensity_ = curr_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                curr_x_ticks = np.arange(min_intensity, max_intensity, self.base)

                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (self.base // 2), right=max_intensity + (self.base // 2))

                """ Iterate over responses """
                j = 0
                for r, response in enumerate(self.response):
                    prefix = f"{tuple(list(curr_combination)[::-1] + [r])}: {response} - MEP"
                    if not j: prefix = curr_combination_inverse + prefix

                    """ MEP Size scatter plot and recruitment curve """
                    postfix = ""
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=self.response_colors[r], ax=ax)
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_mu_map[:, r],
                        ax=ax,
                        **self.recruitment_curve_props,
                    )

                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    """ Observational predictive """
                    postfix = "Prediction"
                    ax = axes[i, j]
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_obs_map[:, r],
                        color="black",
                        ax=ax
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_obs_hpdi_95[0, :, r],
                        curr_obs_hpdi_95[1, :, r],
                        color="C1"
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_obs_hpdi_85[0, :, r],
                        curr_obs_hpdi_85[1, :, r],
                        color="C2"
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_obs_hpdi_65[0, :, r],
                        curr_obs_hpdi_65[1, :, r],
                        color="C3"
                    )
                    sns.scatterplot(
                        data=curr_df,
                        x=self.intensity,
                        y=response,
                        color="yellow",
                        edgecolor="black",
                        ax=ax
                    )

                    ax.sharex(axes[i, 0])
                    ax.sharey(axes[i, j - 1])
                    ax.set_title(postfix)
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    """ Recruitment curve predictive """
                    postfix = "Fit"
                    ax = axes[i, j]
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_mu_map[:, r],
                        color="black",
                        ax=ax
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_mu_hpdi_95[0, :, r],
                        curr_mu_hpdi_95[1, :, r],
                        color="C1"
                    )
                    sns.scatterplot(
                        data=curr_df,
                        x=self.intensity,
                        y=response,
                        color="yellow",
                        edgecolor="black",
                        ax=ax
                    )

                    ax.sharex(axes[i, 0])
                    ax.sharey(axes[i, j - 2])
                    ax.set_title(postfix)
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return

    @timing
    def render_predictive_check(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        prior_predictive: dict | None = None,
        posterior_predictive: dict | None = None,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        destination_path: str | None = None
    ):
        assert (prior_predictive is not None) or (posterior_predictive is not None)

        if posterior_predictive is not None:
            PREDICTIVE = POSTERIOR_PREDICTIVE
            predictive = posterior_predictive
            msg = "Rendering posterior predictive checks ..."
        else:
            PREDICTIVE = PRIOR_PREDICTIVE
            predictive = prior_predictive
            msg = "Rendering prior predictive checks ..."

        if destination_path is None: destination_path = os.path.join(
            self.build_dir, PREDICTIVE
        )
        logger.info(msg)
        return self.predictive_checks_renderer(
            df=df,
            prediction_df=prediction_df,
            predictive=predictive,
            destination_path=destination_path,
            encoder_dict=encoder_dict
        )

    @timing
    def run_trace(self, df: pd.DataFrame):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._collect_regressor(df=df), *self._collect_response(df=df)
            )
        return trace

    @timing
    def run_inference(self, df: pd.DataFrame, sampler: MCMCKernel = None, **kwargs) -> tuple[MCMC, dict]:
        """ Set up sampler """
        if sampler is None: sampler = NUTS(self._model, **kwargs)
        mcmc = MCMC(sampler, **self.mcmc_params)

        """ Run MCMC inference """
        logger.info(f"Running inference with {self.NAME} ...")
        mcmc.run(self.rng_key, *self._collect_regressor(df=df), *self._collect_response(df=df))

        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return mcmc, posterior_samples

    @timing
    def make_prediction_dataset(self, df: pd.DataFrame, num: int = 100):
        pred_df = df \
            .groupby(by=self.combination_columns) \
            .agg({self.intensity: [min, max]}) \
            .copy()
        pred_df.columns = pred_df.columns.map(lambda x: x[1])
        pred_df = pred_df.reset_index().copy()

        pred_df["min"] = pred_df["min"].apply(lambda x: floor(x, base=self.base))
        pred_df["max"] = pred_df["max"] \
            .apply(lambda x: (x, ceil(x, base=self.base))) \
            .apply(lambda x: x[0] + self.base if x[0] == x[1] else x[1])

        pred_df[self.intensity] = pred_df[["min", "max"]] \
            .apply(lambda x: (x[0], x[1], min(2000, ceil((x[1] - x[0]) / 5, base=100))), axis=1) \
            .apply(lambda x: np.linspace(x[0], x[1], num))
        pred_df = pred_df.explode(column=self.intensity)[self.regressors].copy()
        pred_df[self.intensity] = pred_df[self.intensity].astype(float)

        pred_df.reset_index(drop=True, inplace=True)
        return pred_df

    @timing
    def predict(
        self,
        df: pd.DataFrame,
        num_samples: int = 100,
        posterior_samples: dict | None = None
    ):
        if posterior_samples is None:   # Prior predictive
            predictive = Predictive(
                model=self._model, num_samples=num_samples
            )
        else:   # Posterior predictive
            predictive = Predictive(
                model=self._model, posterior_samples=posterior_samples
            )

        """ Generate predictions """
        predictions = predictive(self.rng_key, *self._collect_regressor(df=df))
        predictions = {u: np.array(v) for u, v in predictions.items()}
        return predictions

    @timing
    def save(self, mcmc: numpyro.infer.mcmc.MCMC, **kwargs):
        mcmc_path = kwargs.get(
            "mcmc_path", os.path.join(self.build_dir, MCMC_NC)
        )
        diagnostics_path = kwargs.get(
            "diagnostics_path", os.path.join(self.build_dir, DIAGNOSTICS_CSV)
        )
        loo_path = kwargs.get(
            "loo_path", os.path.join(self.build_dir, LOO_CSV)
        )
        waic_path = kwargs.get(
            "waic_path", os.path.join(self.build_dir, WAIC_CSV)
        )

        """ Save inference data """
        logger.info("Saving inference data ...")
        numpyro_data = az.from_numpyro(mcmc)
        numpyro_data.to_netcdf(mcmc_path)
        logger.info(f"Saved to {mcmc_path}")

        """ Save convergence diagnostics """
        logger.info("Rendering convergence diagnostics ...")
        az.summary(data=numpyro_data, hdi_prob=.95).to_csv(self.diagnostics_path)
        logger.info(f"Saved to {diagnostics_path}")

        """ Model evaluation """
        logger.info("Evaluating model ...")
        score = az.loo(numpyro_data)
        logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
        score.to_csv(loo_path)

        score = az.waic(numpyro_data)
        logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")
        score.to_csv(waic_path)
