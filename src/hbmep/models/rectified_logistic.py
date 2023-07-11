import logging

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import MepConfig
from hbmep.models import Baseline
from hbmep.models.utils import Site as site
from hbmep.utils.constants import RECTIFIED_LOGISTIC

logger = logging.getLogger(__name__)


class RectifiedLogistic(Baseline):
    def __init__(self, config: MepConfig):
        super(RectifiedLogistic, self).__init__(config=config)
        self.link = RECTIFIED_LOGISTIC

        self.mu_a = config.PRIORS[site.mu_a]
        self.sigma_a = config.PRIORS[site.sigma_a]

        self.sigma_b = config.PRIORS[site.sigma_b]

        self.sigma_L = config.PRIORS[site.sigma_L]
        self.sigma_H = config.PRIORS[site.sigma_H]
        self.sigma_v = config.PRIORS[site.sigma_v]

        self.g_1 = config.PRIORS[site.g_1]
        self.g_2 = config.PRIORS[site.g_2]

        self.p = config.PRIORS[site.p]

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.n_subject, n_subject, dim=-2):
                """ Hyper-priors """
                mu_a = numpyro.sample(
                    site.mu_a,
                    dist.TruncatedNormal(self.mu_a[0], self.mu_a[1], low=0)
                )
                sigma_a = numpyro.sample(site.sigma_a, dist.HalfNormal(self.sigma_a))

                sigma_b = numpyro.sample(site.sigma_b, dist.HalfNormal(self.sigma_b))

                sigma_L = numpyro.sample(site.sigma_L, dist.HalfNormal(self.sigma_L))
                sigma_H = numpyro.sample(site.sigma_H, dist.HalfNormal(self.sigma_H))
                sigma_v = numpyro.sample(site.sigma_v, dist.HalfNormal(self.sigma_v))

                with numpyro.plate("n_feature0", n_feature0, dim=-3):
                    """ Priors """
                    a = numpyro.sample(
                        site.a,
                        dist.TruncatedNormal(mu_a, sigma_a, low=0)
                    )
                    b = numpyro.sample(site.b, dist.HalfNormal(sigma_b))

                    L = numpyro.sample(site.L, dist.HalfNormal(sigma_L))
                    H = numpyro.sample(site.H, dist.HalfNormal(sigma_H))
                    v = numpyro.sample(site.v, dist.HalfNormal(sigma_v))

                    g_1 = numpyro.sample(
                        site.g_1, dist.HalfCauchy(self.g_1)
                    )
                    g_2 = numpyro.sample(
                        site.g_2, dist.HalfCauchy(self.g_2)
                    )

                    p = numpyro.sample(site.p, dist.HalfNormal(self.p))

        """ Model """
        mu = numpyro.deterministic(
            site.mu,
            L[feature0, subject] + \
            jnp.maximum(
                0,
                -1 + \
                (H[feature0, subject] + 1) / \
                jnp.power(
                    1 + \
                    (jnp.power(1 + H[feature0, subject], v[feature0, subject]) - 1) * \
                    jnp.exp(-b[feature0, subject] * (intensity - a[feature0, subject])),
                    1 / v[feature0, subject]
                )
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            g_1[feature0, subject] + \
            g_2[feature0, subject] * jnp.power(1 / mu, p[feature0, subject])
        )

        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(mu * beta, beta).to_event(1),
                obs=response_obs
            )
