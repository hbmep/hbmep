import logging

import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi

from hb_mep.config import HBMepConfig
from hb_mep.models.baseline import Baseline
from hb_mep.models.utils import Site as site
from hb_mep.utils import timing

logger = logging.getLogger(__name__)


class BayesianHierarchical(Baseline):
    def __init__(self, config: HBMepConfig):
        super(BayesianHierarchical, self).__init__(config=config)
        self.name = "Bayesian_Hierarchical"

    def model(self, intensity, participant, feature0, response_obs=None):
        n_participant = np.unique(participant).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        with numpyro.plate("n_feature0", n_feature0, dim=-1):
            # Hyperriors
            a_mean = numpyro.sample(site.a_mean, dist.TruncatedDistribution(dist.Normal(150, 50), low=0))
            a_scale = numpyro.sample(site.baseline_scale, dist.HalfNormal(20))

            with numpyro.plate("n_participant", n_participant, dim=-2):
                # Priors
                a = numpyro.sample(
                    site.a,
                    dist.TruncatedDistribution(dist.Normal(a_mean, a_scale), low=0)
                )
                b = numpyro.sample(site.baseline_scale, dist.HalfNormal(20))

                lo = numpyro.sample(site.lo, dist.HalfNormal(0.1))
                g = numpyro.sample(site.g, dist.Beta(1, 24))

                noise_offset = numpyro.sample(site.noise_offset, dist.HalfCauchy(.01))
                noise_slope = numpyro.sample(site.noise_slope, dist.HalfCauchy(.05))

        # Model
        mean = \
            lo[participant, feature0] - \
            jnp.log(jnp.maximum(
                g[participant, feature0],
                jnp.exp(-jax.nn.relu(
                    b[participant, feature0] * (intensity - a[participant, feature0])
                ))
            ))

        noise = \
            noise_offset[participant, feature0] + \
            noise_slope[participant, feature0] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, noise, low=0), obs=response_obs)

    def _get_estimates(
        self,
        posterior_samples: dict,
        posterior_means: dict,
        c: tuple,
        x: np.ndarray
    ):
        a = posterior_means[site.a][c]
        b = posterior_means[site.b][c]
        lo = posterior_means[site.lo][c]
        y = lo - jnp.log(jnp.maximum(g, jnp.exp(-jnp.maximum(0, b * (x - a)))))

        threshold_samples = posterior_samples[site.a][:, c[0], c[1]]
        hpdi_interval = hpdi(threshold_samples, prob=0.95)

        return y, threshold_samples, hpdi_interval
