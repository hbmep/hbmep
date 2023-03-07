import os
import logging
from pathlib import Path

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import h5py
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from hb_mep.config import HBMepConfig
from hb_mep.utils.constants import (
    REPORTS_DIR,
    INTENSITY,
    RESPONSE_MUSCLES,
    PARTICIPANT,
    INDEPENDENT_FEATURES
)

logger = logging.getLogger(__name__)


class Baseline():
    def __init__(self, config: HBMepConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

        self.name = 'baseline'
        self.link = jax.nn.relu

        self.random_state = 0

    def model(self, intensity, participant, independent, response_obs=None):
        a_level_scale_global_scale = numpyro.sample('a_global_scale', dist.HalfNormal(2.0))
        a_level_mean_global_scale = numpyro.sample('a_level_mean_global_scale', dist.HalfNormal(5.0))

        b_level_mean_global_scale = numpyro.sample('b_level_mean_global_scale', dist.HalfNormal(5.0))
        b_level_scale_global_scale = numpyro.sample('b_global_scale', dist.HalfNormal(2.0))

        lo_level_mean_global_scale = numpyro.sample('lo_level_mean_global_scale', dist.HalfNormal(2.0))
        lo_level_scale_global_scale = numpyro.sample('lo_level_scale_global_scale', dist.HalfNormal(2.0))

        sigma_offset_level_scale_global_scale = \
            numpyro.sample('sigma_offset_level_scale_global_scale', dist.HalfCauchy(5.0))
        sigma_slope_level_scale_global_scale = \
            numpyro.sample('sigma_slope_level_scale_global_scale', dist.HalfCauchy(5.0))

        n_participants = np.unique(participant).shape[0]
        n_levels = np.unique(independent).shape[0]

        with numpyro.plate("n_levels", n_levels, dim=-2):
            a_level_mean = numpyro.sample("a_level_mean", dist.HalfNormal(a_level_mean_global_scale))
            b_level_mean = numpyro.sample("b_level_mean", dist.HalfNormal(b_level_mean_global_scale))

            a_level_scale = numpyro.sample("a_level_scale", dist.HalfNormal(a_level_scale_global_scale))
            b_level_scale = numpyro.sample("b_level_scale", dist.HalfNormal(b_level_scale_global_scale))

            lo_level_mean = numpyro.sample("lo_level_mean", dist.HalfNormal(lo_level_mean_global_scale))
            lo_level_scale = numpyro.sample("lo_level_scale", dist.HalfNormal(lo_level_scale_global_scale))

            sigma_offset_level_scale = \
                numpyro.sample(
                    'sigma_offset_level_scale',
                    dist.HalfCauchy(sigma_offset_level_scale_global_scale)
                )
            sigma_slope_level_scale = \
                numpyro.sample(
                    'sigma_slope_level_scale',
                    dist.HalfCauchy(sigma_slope_level_scale_global_scale)
                )

            with numpyro.plate("n_participants", n_participants, dim=-1):
                a = numpyro.sample("a", dist.Normal(a_level_mean, a_level_scale))
                b = numpyro.sample("b", dist.Normal(b_level_mean, b_level_scale))

                lo = numpyro.sample("lo", dist.Normal(lo_level_mean, lo_level_scale))

                sigma_offset = numpyro.sample('sigma_offset', dist.HalfCauchy(sigma_offset_level_scale))
                sigma_slope = numpyro.sample('sigma_slope', dist.HalfCauchy(sigma_slope_level_scale))

        mean = lo[independent, participant] + self.link(b[independent, participant] * (intensity - a[independent, participant]))
        sigma = sigma_offset[independent, participant] + sigma_slope[independent, participant] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=response_obs)

    # def render(
    #     self,
    #     data_dict: dict
    #     ) -> graphviz.graphs.Digraph:
    #     """
    #     Render NumPyro model and save resultant graph.

    #     Args:
    #         model (model): NumPyro model for rendering.
    #         data_dict (dict): Data dictionary containing model parameters for rendering.
    #         filename (Optional[Path], optional): Target destination for saving rendered graph. Defaults to None.

    #     Returns:
    #         graphviz.graphs.Digraph: Rendered graph.
    #     """
    #     logger.info('Rendering model ...')
    #     # Retrieve data from data dictionary for rendering model
    #     intensity, mep_size, participant, segment  = \
    #         itemgetter(INTENSITY, MEP_SIZE, PARTICIPANT, SEGMENT)(data_dict)
    #     return numpyro.render_model(
    #         self.model,
    #         model_args=(intensity, participant, segment, mep_size),
    #         filename=os.path.join(self.reports_path, self.config.RENDER_FNAME)
    #     )

    def sample(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference

        Args:
            data_dict (dict): Data dictionary containing input and observations

        Returns:
            tuple[numpyro.infer.mcmc.MCMC, dict]: MCMC inference results and posterior samples.
        """
        response = df[RESPONSE_MUSCLES].to_numpy().reshape(-1,)
        participant = df[PARTICIPANT].to_numpy().reshape(-1,)
        independent = df[INDEPENDENT_FEATURES].to_numpy().reshape(-1,)
        intensity = df[INTENSITY].to_numpy().reshape(-1,)

        # MCMC
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info(f'Running inference with model {self.name}...')
        mcmc.run(rng_key, intensity, participant, independent, response)
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples

    def plot_fit(
            self,
            df: pd.DataFrame,
            posterior_samples: dict
    ):
        n_muscles = 1

        combinations = \
            df \
            .groupby(by=[PARTICIPANT] + INDEPENDENT_FEATURES) \
            .size() \
            .to_frame('counts') \
            .reset_index().copy()
        combinations = combinations[[PARTICIPANT] + INDEPENDENT_FEATURES].apply(tuple, axis=1).tolist()
        total_combinations = len(combinations)

        fig, axes = plt.subplots(total_combinations*n_muscles, 3, figsize=(12,total_combinations*n_muscles*5))

        mean_a = posterior_samples['a'].mean(axis=0)
        mean_b = posterior_samples['b'].mean(axis=0)

        if 'lo' in posterior_samples:
            mean_lo = posterior_samples['lo'].mean(axis=0)

        if 'hi' in posterior_samples:
            mean_hi = posterior_samples['hi'].mean(axis=0)

        for i, c in enumerate(combinations):
            temp = \
                df[df[[PARTICIPANT] + INDEPENDENT_FEATURES] \
                .apply(tuple, axis=1) \
                .isin([c])] \
                .reset_index(drop=True) \
                .copy()

            a = mean_a[c[::-1]]
            b = mean_b[c[::-1]]

            if 'lo' in posterior_samples:
                lo = mean_lo[c[::-1]]

            if 'hi' in posterior_samples:
                hi = mean_hi[c[::-1]]

            axes[i, 0].set_title(f'Actual: Combination:{c}, {RESPONSE_MUSCLES[0]}')
            axes[i, 0].set_title(f'Actual: Combination:{c}, {RESPONSE_MUSCLES[0]}')

            sns.scatterplot(data=temp, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i, 0])
            sns.scatterplot(data=temp, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i, 1], alpha=.4)
            sns.scatterplot(data=temp, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i, 2])

            x_val = np.linspace(0, 15, 100)
            y_val = self.link(b * (x_val - a))

            if 'lo' in posterior_samples:
                y_val += lo

            sns.kdeplot(x=posterior_samples['a'][:,c[-1],c[-2]], ax=axes[i, 1], color='green')
            sns.lineplot(
                x=x_val,
                y=y_val,
                ax=axes[i, 1],
                color='red',
                alpha=0.4,
                label=f'Mean Posterior {RESPONSE_MUSCLES[0]}'
            )
            sns.lineplot(
                x=x_val,
                y=y_val,
                ax=axes[i, 2],
                color='red',
                alpha=0.4,
                label=f'Mean Posterior {RESPONSE_MUSCLES[0]}'
            )

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
        return fig

    def plot_kde(self, df: pd.DataFrame, posterior_samples: dict):
        fig, ax = plt.subplots(df[PARTICIPANT].nunique(), 1)

        combinations = \
            df \
            .groupby(by=[PARTICIPANT] + INDEPENDENT_FEATURES) \
            .size() \
            .to_frame('counts') \
            .reset_index().copy()
        combinations = combinations[[PARTICIPANT] + INDEPENDENT_FEATURES].apply(tuple, axis=1).tolist()

        for i, c in enumerate(combinations):
            sns.kdeplot(posterior_samples['a'][:,c[-1],c[-2]], label=f'{c[-1]}', ax=ax)
            ax.set_title(f'Participant: {c[0]} - {RESPONSE_MUSCLES[0]}')
            ax.set_xlim(left=0)
        plt.legend();
        return fig