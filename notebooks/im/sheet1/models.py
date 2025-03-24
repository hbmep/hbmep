import numpy as np
from jax import numpy as jnp
import numpyro as pyro
from numpyro import distributions as dist

from hbmep.model import BaseModel

import functional as F
from utils import Site as site


class ImmunoModel(BaseModel):
    def __init__(self, *args, **kw):
        super(ImmunoModel, self).__init__(*args, **kw)
        self.range_restricted = False
        self.intensity = "conc"
        self.features = ["contam"]
        self.response = ["od"]
        self.mcmc_params = {
            "num_warmup": 4000,
            "num_samples": 1000,
            "num_chains": 4,
            "thinning": 1,
        }
        self.nuts_params = {
            "max_tree_depth": (20, 20),
            "target_accept_prob": .95,
        }

    def nhb_l4(self, intensity, features, response=None, **kw):
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                b3 = pyro.sample(site.b3, dist.Normal(5, 5))
                b4 = pyro.sample(site.b4, dist.HalfNormal(5.))

                b1 = pyro.sample(site.b1, dist.HalfNormal(.1))
                b2 = pyro.sample(site.b2, dist.HalfNormal(5.))

                c1 = pyro.sample(site.c1, dist.HalfNormal(5.))
                c2 = pyro.sample(site.c2, dist.HalfNormal(.5))

        # Model
        mu, alpha, beta = self.gamma_likelihood(
            F.logistic4,
            intensity,
            (
                b3[feature0],
                b4[feature0],
                b1[feature0],
                b2[feature0],
            ),
            c1[feature0],
            c2[feature0]
        )
        pyro.deterministic(site.mu, mu)
        # Observation
        pyro.sample(
            site.obs,
            dist.Gamma(concentration=alpha, rate=beta),
            obs=response
        )

    def hb1_l4(self, intensity, features, response=None, **kw):
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        b3_loc = pyro.sample(site.b3.loc, dist.Normal(5., 5.))
        b3_scale = pyro.sample(site.b3.scale, dist.HalfNormal(5.))

        b4_scale = pyro.sample(site.b4.scale, dist.HalfNormal(5.))
        b1_scale = pyro.sample(site.b1.scale, dist.HalfNormal(.1))

        b2_loc = pyro.sample(site.b2.loc, dist.TruncatedNormal(2., 5., low=0))
        b2_scale = pyro.sample(site.b2.scale, dist.HalfNormal(5.))
        
        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                b3_raw = pyro.sample(site.b3.raw, dist.Normal(0, 1))
                b3 = pyro.deterministic(site.b3, b3_loc + b3_scale * b3_raw)

                b4_raw = pyro.sample(site.b4.raw, dist.HalfNormal(1.))
                b4 = pyro.deterministic(site.b4, b4_scale * b4_raw)

                b1_raw = pyro.sample(site.b1.raw, dist.HalfNormal(1.))
                b1 = pyro.deterministic(site.b1, b1_scale * b1_raw)

                b2 = pyro.sample(site.b2, dist.TruncatedNormal(b2_loc, b2_scale, low=0))

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1.))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1.))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Model
        mu, alpha, beta = self.gamma_likelihood(
            F.logistic4,
            intensity,
            (
                b3[feature0],
                b4[feature0],
                b1[feature0],
                b2[feature0],
            ),
            c1[feature0],
            c2[feature0]
        )
        pyro.deterministic(site.mu, mu)
        # Observation
        pyro.sample(
            site.obs,
            dist.Gamma(concentration=alpha, rate=beta),
            obs=response
        )

    def nhb_ln(self, intensity, features, response=None, **kw):
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                # Baseline
                b3 = pyro.sample(site.b3, dist.Normal(5, 5))
                b4 = pyro.sample(site.b4, dist.HalfNormal(5.))

                b1 = pyro.sample(site.b1, dist.HalfNormal(.1))
                b2 = pyro.sample(site.b2, dist.HalfNormal(5.))

                # alpha = pyro.sample(site.alpha, dist.HalfNormal(1.))
                sigma = pyro.sample(site.sigma, dist.HalfNormal(5))

        # Model
        mu = pyro.deterministic(
            site.mu,
            F.logistic4(
                intensity,
                b3[feature0],
                b4[feature0],
                b1[feature0],
                b2[feature0],
            )
        )
        loc = jnp.log(mu)
        # loc = mu
        scale = sigma[feature0]

        # Observation
        pyro.sample(
            "obs",
            dist.Normal(loc=loc, scale=scale),
            obs=jnp.log(response) if response is not None else None
        )

    def nhb_ln_r01(self, intensity, features, response=None, **kw):
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                # Baseline
                b3 = pyro.sample(site.b3, dist.Normal(5, 5))
                b4 = pyro.sample(site.b4, dist.HalfNormal(5.))

                b1 = pyro.sample(site.b1, dist.HalfNormal(.1))
                b2 = pyro.sample(site.b2, dist.HalfNormal(5.))

                # alpha = pyro.sample(site.alpha, dist.HalfNormal(1.))
                sigma = pyro.sample(site.sigma, dist.HalfNormal(5))

        # Model
        mu = pyro.deterministic(
            site.mu,
            F.logistic4(
                intensity,
                b3[feature0],
                b4[feature0],
                b1[feature0],
                b2[feature0],
            )
        )
        loc = jnp.log(mu)
        # loc = mu
        scale = sigma[feature0]

        # Observation
        pyro.sample(
            "obs",
            dist.Normal(loc=loc, scale=scale),
            obs=jnp.log(response) if response is not None else None
        )
