import logging

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist

from hb_mep.config import HBMepConfig
from hb_mep.models.baseline import Baseline

logger = logging.getLogger(__name__)


class LogisticRegression(Baseline):
    def __init__(self, config: HBMepConfig):
        super(LogisticRegression, self).__init__(config=config)
        self.name = 'logistic_regression'
        self.link = jax.nn.sigmoid

    def model(self, intensity, participant, segment, mep_size_obs=None):
        a_level_mean_global_scale = numpyro.sample('a_level_mean_global_scale', dist.HalfNormal(1.0))
        b_level_mean_global_scale = numpyro.sample('b_level_mean_global_scale', dist.HalfNormal(1.0))

        a_level_scale_global_scale = numpyro.sample('a_global_scale', dist.HalfNormal(1.0))
        b_level_scale_global_scale = numpyro.sample('b_global_scale', dist.HalfNormal(1.0))

        n_participants = np.unique(participant).shape[0]
        n_levels = np.unique(segment).shape[0]

        with numpyro.plate("n_levels", n_levels, dim=-2):
            a_level_mean = numpyro.sample("a_level_mean", dist.HalfNormal(a_level_mean_global_scale))
            b_level_mean = numpyro.sample("b_level_mean", dist.HalfNormal(b_level_mean_global_scale))

            a_level_scale = numpyro.sample("a_level_scale", dist.HalfNormal(a_level_scale_global_scale))
            b_level_scale = numpyro.sample("b_level_scale", dist.HalfNormal(b_level_scale_global_scale))

            with numpyro.plate("n_participants", n_participants, dim=-1):
                a = numpyro.sample("a", dist.Normal(a_level_mean, a_level_scale))
                b = numpyro.sample("b", dist.Normal(b_level_mean, b_level_scale))

        mean = self.link(b[segment, participant] * (intensity - a[segment, participant]))

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.Bernoulli(mean), obs=mep_size_obs)
