import pandas as pd
import numpy as np
import jax
from jax import numpy as jnp
import numpyro as pyro
from numpyro import distributions as dist

from hbmep.model import BaseModel, NonHierarchicalBaseModel
from hbmep.notebooks.rat.util import get_subname

import functional as F
from util import Site as site


class nHB(NonHierarchicalBaseModel):
    def __init__(self, *args, **kw):
        super(nHB, self).__init__(*args, **kw)
        self.n_jobs = -1
        self.test_run = False
        self.use_mixture = False
        self.b3_var = None
        self.run_id = None

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def non_hierarchical(self, intensity, features, response=None, **kw):
        intensity = intensity[..., 0]

        b1_log = pyro.sample(site.b1.log, dist.Normal(0, 10))
        b1 = pyro.deterministic(site.b1, jnp.exp(b1_log))

        b2_log = pyro.sample(site.b2.log, dist.Normal(0, 10))
        b2 = pyro.deterministic(site.b2, jnp.exp(b2_log))

        b3_log = pyro.sample(site.b3.log, dist.Normal(0, 10))
        b3 = pyro.deterministic(site.b3, jnp.exp(b3_log))

        b4_log = pyro.sample(site.b4.log, dist.Normal(0, 10))
        b4 = pyro.deterministic(site.b4, jnp.exp(b4_log))

        sigma = pyro.sample(site.sigma, dist.HalfNormal(10.))

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        mu = pyro.deterministic(site.mu, F.ro1(intensity, b3, b4, b1, b2))
        loc = jnp.log(mu)
        scale = sigma

        if self.use_mixture:
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions=[
                dist.Normal(loc=loc, scale=scale),
                dist.Normal(loc=0, scale=10.)
            ]
            Mixture = dist.MixtureGeneral(
                mixing_distribution=mixing_distribution,
                component_distributions=component_distributions
            )

        obs_log = pyro.sample(
            site.obs.log,
            (
                Mixture if self.use_mixture
                else dist.Normal(loc=loc, scale=scale)
            ),
            obs=jnp.log(response) if response is not None else None
        )
        y_ = pyro.deterministic(site.obs, jnp.exp(obs_log))
        # if self.use_mixture:
        #     log_probs = Mixture.component_log_probs(y_)
        #     pyro.deterministic(
        #         "obs_outlier_prob", log_probs - jax.nn.logsumexp(log_probs, axis=-1, keepdims=True)
        #     )
