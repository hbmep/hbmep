import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hb_mep.config import HBMepConfig
from hb_mep.models.baseline import Baseline
from hb_mep.utils.constants import (
    INTENSITY,
    RESPONSE_MUSCLES,
    PARTICIPANT,
    INDEPENDENT_FEATURES
)

logger = logging.getLogger(__name__)


class SaturatedExponential(Baseline):
    def __init__(self, config: HBMepConfig):
        super(SaturatedExponential, self).__init__(config=config)
        self.name = 'saturated_exponential'
        self.link = jax.nn.sigmoid

    def model(self, intensity, participant, independent, response_obs=None):
        a_level_scale_global_scale = numpyro.sample('a_global_scale', dist.HalfNormal(2.0))
        a_level_mean_global_scale = numpyro.sample('a_level_mean_global_scale', dist.HalfNormal(5.0))

        b_level_mean_global_scale = numpyro.sample('b_level_mean_global_scale', dist.HalfNormal(5.0))
        b_level_scale_global_scale = numpyro.sample('b_global_scale', dist.HalfNormal(2.0))

        lo_level_mean_global_scale = numpyro.sample('lo_level_mean_global_scale', dist.HalfNormal(2.0))
        lo_level_scale_global_scale = numpyro.sample('lo_level_scale_global_scale', dist.HalfNormal(2.0))

        hi_level_mean_global_scale = numpyro.sample('hi_level_mean_global_scale', dist.HalfNormal(5.0))
        hi_level_scale_global_scale = numpyro.sample('hi_level_scale_global_scale', dist.HalfNormal(2.0))

        sigma_offset_level_scale_global_scale = \
            numpyro.sample('sigma_offset_level_scale_global_scale', dist.HalfCauchy(5.0))
        sigma_slope_level_scale_global_scale = \
            numpyro.sample('sigma_slope_level_scale_global_scale', dist.HalfCauchy(5.0))

        n_participant = np.unique(participant).shape[0]
        n_independent = np.unique(independent).shape[0]

        with numpyro.plate("n_independent", n_independent, dim=-1):
            a_level_mean = numpyro.sample("a_level_mean", dist.HalfNormal(a_level_mean_global_scale))
            a_level_scale = numpyro.sample("a_level_scale", dist.HalfNormal(a_level_scale_global_scale))

            b_level_mean = numpyro.sample("b_level_mean", dist.HalfNormal(b_level_mean_global_scale))
            b_level_scale = numpyro.sample("b_level_scale", dist.HalfNormal(b_level_scale_global_scale))

            lo_level_mean = numpyro.sample("lo_level_mean", dist.HalfNormal(lo_level_mean_global_scale))
            lo_level_scale = numpyro.sample("lo_level_scale", dist.HalfNormal(lo_level_scale_global_scale))

            hi_level_mean = numpyro.sample("hi_level_mean", dist.HalfNormal(hi_level_mean_global_scale))
            hi_level_scale = numpyro.sample("hi_level_scale", dist.HalfNormal(hi_level_scale_global_scale))

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

            with numpyro.plate("n_participant", n_participant, dim=-2):
                a = numpyro.sample("a", dist.Normal(a_level_mean, a_level_scale))
                b = 5 + numpyro.sample("b", dist.Normal(b_level_mean, b_level_scale))

                lo = numpyro.sample("lo", dist.Normal(lo_level_mean, lo_level_scale))
                hi = numpyro.sample("hi", dist.Normal(hi_level_mean, hi_level_scale))

                sigma_offset = numpyro.sample('sigma_offset', dist.HalfCauchy(sigma_offset_level_scale))
                sigma_slope = numpyro.sample('sigma_slope', dist.HalfCauchy(sigma_slope_level_scale))

        mean = jnp.maximum(
            hi[participant, independent] - \
            (hi[participant, independent] - lo[participant, independent]) * jax.numpy.exp(
                - b[participant, independent] * (intensity - a[participant, independent])
            ),
            lo[participant, independent]
        )
        sigma = sigma_offset[participant, independent] + sigma_slope[participant, independent] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=response_obs)

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

            a = mean_a[c[::]]
            b = 5 + mean_b[c[::]]

            if 'lo' in posterior_samples:
                lo = mean_lo[c[::]]

            if 'hi' in posterior_samples:
                hi = mean_hi[c[::]]

            axes[i, 0].set_title(f'Actual: Combination:{c}, {RESPONSE_MUSCLES[0]}')
            axes[i, 1].set_title(f'Fitted: Combination:{c}, {RESPONSE_MUSCLES[0]}')

            sns.scatterplot(data=temp, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i, 0])
            sns.scatterplot(data=temp, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i, 1], alpha=.4)
            sns.scatterplot(data=temp, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i, 2])

            x_val = np.linspace(0, 15, 100)
            y_val = jnp.maximum(hi - (hi - lo) * jax.numpy.exp(-b * (x_val - a)), lo)

            sns.kdeplot(x=posterior_samples['a'][:,c[-2],c[-1]], ax=axes[i, 1], color='green')
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
            axes[i, 1].set_ylim(bottom=0, top=temp[RESPONSE_MUSCLES[0]].max() + 5)
            axes[i, 2].set_ylim(bottom=0, top=temp[RESPONSE_MUSCLES[0]].max() + 5)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
        return fig
