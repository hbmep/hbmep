import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from hbmep.config import HBMepConfig
from hbmep.models.baseline import Baseline

logger = logging.getLogger(__name__)


class MVBaseline(Baseline):
    def __init__(self, config: HBMepConfig):
        super(MVBaseline, self).__init__(config=config)
        self.name = 'mv_baseline'
        self.link = jax.nn.relu

    def model(self, intensity, participant, independent, response_obs=None):
        a_level_mean_shared_mean = numpyro.sample('a_level_mean_shared_mean', dist.HalfNormal(10.0))
        a_level_scale_global_scale = numpyro.sample('a_global_scale', dist.HalfNormal(2.0))

        b_level_mean_global_scale = numpyro.sample('b_level_mean_global_scale', dist.HalfNormal(5.0))
        b_level_scale_global_scale = numpyro.sample('b_global_scale', dist.HalfNormal(2.0))

        lo_level_mean_global_scale = numpyro.sample('lo_level_mean_global_scale', dist.HalfNormal(2.0))
        lo_level_scale_global_scale = numpyro.sample('lo_level_scale_global_scale', dist.HalfNormal(2.0))

        sigma_offset_level_scale_global_scale = \
            numpyro.sample('sigma_offset_level_scale_global_scale', dist.HalfCauchy(5.0))
        sigma_slope_level_scale_global_scale = \
            numpyro.sample('sigma_slope_level_scale_global_scale', dist.HalfCauchy(5.0))

        a_level_mean = numpyro.sample(
            "a_level_mean",
            dist.MultivariateNormal(
                a_level_mean_shared_mean * jnp.ones(5), 10 * jnp.diag(jnp.ones(5))
            )
        )

        n_participant = np.unique(participant).shape[0]
        n_independent = np.unique(independent).shape[0]

        with numpyro.plate("n_independent", n_independent, dim=-1):
            a_level_scale = numpyro.sample("a_level_scale", dist.HalfNormal(a_level_scale_global_scale))

            b_level_mean = numpyro.sample("b_level_mean", dist.HalfNormal(b_level_mean_global_scale))
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

            with numpyro.plate("n_participant", n_participant, dim=-2):
                a = numpyro.sample("a", dist.Normal(a_level_mean, a_level_scale))
                b = numpyro.sample("b", dist.Normal(b_level_mean, b_level_scale))

                lo = numpyro.sample("lo", dist.Normal(lo_level_mean, lo_level_scale))

                sigma_offset = numpyro.sample('sigma_offset', dist.HalfCauchy(sigma_offset_level_scale))
                sigma_slope = numpyro.sample('sigma_slope', dist.HalfCauchy(sigma_slope_level_scale))

        mean = lo[participant, independent] + self.link(
            jnp.multiply(b[participant, independent], intensity - a[participant, independent])
        )
        sigma = sigma_offset[participant, independent] + sigma_slope[participant, independent] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=response_obs)
