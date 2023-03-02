import os
import logging
from pathlib import Path
from operator import itemgetter
from typing import Optional

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import h5py
import graphviz
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from hb_mep.config import HBMepConfig
from hb_mep.utils.constants import (
    REPORTS_DIR,
    NUM_PARTICIPANTS,
    NUM_SEGMENTS,
    TOTAL_COMBINATIONS,
    SEGMENTS_PER_PARTICIPANT,
    INTENSITY,
    MEP_SIZE,
    PARTICIPANT,
    SEGMENT,
    PARTICIPANT_ENCODER,
    SEGMENT_ENCODER
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
        numpyro.set_platform('cpu')
        numpyro.set_host_device_count(4)

    def model(self, intensity, participant, segment, mep_size_obs=None):
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
        n_levels = np.unique(segment).shape[0]

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

                sigma_offset = numpyro.sample('sigma_offset', dist.HalfCauchy(sigma_offset_level_scale))
                sigma_slope = numpyro.sample('sigma_slope', dist.HalfCauchy(sigma_slope_level_scale))

        mean = lo[segment, participant] + self.link(b[segment, participant] * (intensity - a[segment, participant]))
        sigma = sigma_offset[segment, participant] + sigma_slope[segment, participant] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=mep_size_obs)

    def render(
        self,
        data_dict: dict
        ) -> graphviz.graphs.Digraph:
        """
        Render NumPyro model and save resultant graph.

        Args:
            model (model): NumPyro model for rendering.
            data_dict (dict): Data dictionary containing model parameters for rendering.
            filename (Optional[Path], optional): Target destination for saving rendered graph. Defaults to None.

        Returns:
            graphviz.graphs.Digraph: Rendered graph.
        """
        logger.info('Rendering model ...')
        # Retrieve data from data dictionary for rendering model
        intensity, mep_size, participant, segment  = \
            itemgetter(INTENSITY, MEP_SIZE, PARTICIPANT, SEGMENT)(data_dict)
        return numpyro.render_model(
            self.model,
            model_args=(intensity, participant, segment, mep_size),
            filename=os.path.join(self.reports_path, self.config.RENDER_FNAME)
        )

    def sample(self, data_dict: dict) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference

        Args:
            data_dict (dict): Data dictionary containing input and observations

        Returns:
            tuple[numpyro.infer.mcmc.MCMC, dict]: MCMC inference results and posterior samples.
        """
        # Retrieve data from data dictionary
        intensity, mep_size, participant, segment  = \
            itemgetter(INTENSITY, MEP_SIZE, PARTICIPANT, SEGMENT)(data_dict)

        # MCMC
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info(f'Running inference with model {self.name}...')
        mcmc.run(rng_key, intensity, participant, segment, mep_size)
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples

    def plot_fit(
            self,
            df: pd.DataFrame,
            data_dict: dict,
            encoders_dict: dict[str, LabelEncoder],
            posterior_samples: dict,
            keep_muscles: list[str] = ['Biceps'],
            model_function: str = 'relu',
            plot_threshold_kde: bool = True,
            gt: Optional[h5py._hl.files.File] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot inference results

        Args:
            df (pd.DataFrame): Data with
            encoders_dict (dict[str, LabelEncoder]): Encoders for participants and levels.
            data_dict (dict): Data dictionary from preprocess.
            posterior_samples (dict): Posterior samples from MCMC.
            keep_muscles (list[str], optional): Plots for only these muscles will be included. Defaults to ['Biceps'].
            model_function (str, optional): Model function to fit on posterior_samples. Must be 'relu', 'sigmoid', or 'softplus'. Defaults to 'relu'.
            plot_threshold_kde (bool, optional): Whether to plot kernel density plot for threshold from posterior_samples. Defaults to True.
            gt (Optional[h5py._hl.files.File], optional): Ground truth for threshold, used when working with simulated data. Defaults to None.

        Returns:
            matplotlib.figure.Figure: Figure with subplots.
        """
        # Validate arguments
        assert(model_function in ['relu', 'sigmoid', 'softplus'])

        df[keep_muscles] = pd.DataFrame(df[MEP_SIZE].to_list(), columns=keep_muscles)

        n_muscles = len(keep_muscles)
        colors = ['red', 'green', 'blue', 'magenta']
        colors = colors[:n_muscles]

        participant_encoder, level_encoder = itemgetter(PARTICIPANT_ENCODER, SEGMENT_ENCODER)(encoders_dict)
        n_participants, n_levels, levels_per_participant, total_combinations = itemgetter(NUM_PARTICIPANTS, NUM_SEGMENTS, SEGMENTS_PER_PARTICIPANT, TOTAL_COMBINATIONS)(data_dict)

        if plot_threshold_kde:
            fig, axes = plt.subplots(total_combinations*n_muscles, 3, figsize=(12,total_combinations*n_muscles*5))
        else:
            fig, axes = plt.subplots(total_combinations*n_muscles, 2, figsize=(12,total_combinations*n_muscles*5))

        if gt:
            th = np.array(gt['th']).T
            sl = np.array(gt['slope']).T

        posterior_samples['a'] = posterior_samples['a'].reshape(-1, n_levels, n_participants, n_muscles)
        posterior_samples['b'] = posterior_samples['b'].reshape(-1, n_levels, n_participants, n_muscles)

        mean_a = posterior_samples['a'].mean(axis=0)
        mean_b = posterior_samples['b'].mean(axis=0)

        if 'lo' in posterior_samples:
            posterior_samples['lo'] = posterior_samples['lo'].reshape(-1, n_levels, n_participants, n_muscles)
            mean_lo = posterior_samples['lo'].mean(axis=0)

        k = 0
        for i in range(n_participants):
            for j in levels_per_participant[i]:
                for m, col in enumerate(keep_muscles):
                    if participant_encoder is not None and level_encoder is not None:
                        axes[k+m, 0].set_title(f'Actual: Participant:{i}, Level:{j}, {col}')
                        axes[k+m, 1].set_title(f'Fitted: Participant:{i}, Level:{j}, {col}')
                    else:
                        axes[k+m, 0].set_title(f'Actual: Participant:{i+1}, Level:{j+1}')
                        axes[k+m, 1].set_title(f'Fitted: Participant:{i}, Level:{j}')

                    sns.scatterplot(data=df[(df.participant==i) & (df.level==j)], x='intensity', y='raw_mep_size', ax=axes[k+m, 0])
                    sns.scatterplot(data=df[(df.participant==i) & (df.level==j)], x='intensity', y=col, ax=axes[k+m, 1], alpha=.4)

                    x_val = np.linspace(0, 15, 100)
                    y_val = self.link(mean_b[j,i,m] * (x_val - mean_a[j,i,m]))

                    if 'lo' in posterior_samples:
                        y_val += mean_lo[j,i,m]

                    sns.lineplot(x=x_val, y=y_val, ax=axes[k+m, 1], color=colors[m], alpha=0.4, label=f'Mean Posterior {col}')

                    if plot_threshold_kde:
                        sns.scatterplot(data=df[(df.participant==i) & (df.level==j)], x='intensity', y=col, ax=axes[k+m, 2])
                        sns.kdeplot(x=posterior_samples['a'][:,j,i,m], ax=axes[k+m, 1], color='green')
                        sns.lineplot(x=x_val, y=y_val, ax=axes[k+m, 2], color=colors[m], alpha=0.4, label=f'Mean Posterior {col}')

                    if gt:
                        y_val_gt = jax.nn.relu(sl[i,j] * (x_val - th[i,j]))
                        sns.lineplot(x=x_val, y=y_val, ax=axes[k+m, 2], color=colors[m], alpha=0.4, label='GT')

                k += n_muscles

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
        return fig

    def plot_kde(self, data_dict: dict, posterior_samples: dict):
        fig, ax = plt.subplots(data_dict[NUM_PARTICIPANTS], 1)

        for participant in range(data_dict[NUM_PARTICIPANTS]):
            for segment in range(data_dict[NUM_SEGMENTS]):
                sns.kdeplot(posterior_samples['a'][:, segment, 0, 0], label=f'{segment}', ax=ax)
            ax.set_title(f'Participant: {participant} - {MEP_SIZE}')
            ax.set_xlim(left=0)
        plt.legend();
        return fig