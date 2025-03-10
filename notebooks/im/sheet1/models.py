import numpy as np
import numpyro as pyro
from numpyro import distributions as dist

from hbmep.model import BaseModel
from hbmep import functional as F
from hbmep.util import site


class ImmunoModel(BaseModel):
    def __init__(self, *args, **kw):
        super(ImmunoModel, self).__init__(*args, **kw)
        self.intensity = "conc"
        self.features = [["plate", "dilution"]]
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

    def nhb_logistic4(self, intensity, features, response=None, **kw):
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                a = pyro.sample(site.a, dist.Normal(5, 5))
                b = pyro.sample(site.b, dist.HalfNormal(5.))

                g = pyro.sample(site.g, dist.HalfNormal(.1))
                h = pyro.sample(site.h, dist.HalfNormal(5.))

                c1 = pyro.sample(site.c1, dist.HalfNormal(5.))
                c2 = pyro.sample(site.c2, dist.HalfNormal(.5))

        # Model
        mu, alpha, beta = self.gamma_likelihood(
            F.logistic4,
            intensity,
            (
                a[feature0],
                b[feature0],
                g[feature0],
                h[feature0],
            ),
            c1[feature0],
            c2[feature0]
        )
        pyro.deterministic(site.mu, mu)
        # Observation
        pyro.sample(
            "obs",
            dist.Gamma(concentration=alpha, rate=beta),
            obs=response
        )
