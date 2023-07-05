import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi

from hbmep.config import HBMepConfig
from hbmep.models.baseline import Baseline
from hbmep.models.utils import Site as site
from hbmep.utils import timing
from hbmep.utils.constants import (
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


class GammaRegression(Baseline):
    def __init__(self, config: HBMepConfig):
        super(GammaRegression, self).__init__(config=config)
        self.name = "Gamma_Regression"

        self.columns = [PARTICIPANT, FEATURES[1]]
        self.x = np.linspace(0, 15, 100)
        self.xpad = 2

    def _model(self, intensity, participant, feature0, feature1, response_obs=None):
        n_data = intensity.shape[0]
        n_participant = np.unique(participant).shape[0]
        n_feature1 = np.unique(feature1).shape[0]

        with numpyro.plate("n_participant", n_participant, dim=-1):
            """ Hyper-priors """
            a_mean = numpyro.sample(
                site.a_mean,
                dist.TruncatedNormal(5, 10, low=0)
            )
            a_scale = numpyro.sample(site.a_scale, dist.HalfNormal(5))

            b_mean = numpyro.sample(
                "b_mean",
                dist.TruncatedNormal(10, 5, low=0)
            )
            b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(20))

            h_mean = numpyro.sample(
                "h_mean",
                dist.TruncatedNormal(15, 10, low=0)
            )
            h_scale = numpyro.sample("h_scale", dist.HalfNormal(20))

            v_mean = numpyro.sample(
                "v_mean",
                dist.TruncatedNormal(50, 30, low=0)
            )
            v_scale = numpyro.sample("v_scale", dist.HalfNormal(20))

            lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(2))

            with numpyro.plate("n_feature1", n_feature1, dim=-2):
                """ Priors """
                a = numpyro.sample(
                    site.a,
                    dist.TruncatedNormal(a_mean, a_scale, low=0)
                )
                b = numpyro.sample(
                    site.b,
                    dist.TruncatedNormal(b_mean, b_scale, low=0)
                )

                h = numpyro.sample(
                    "h",
                    dist.TruncatedNormal(h_mean, h_scale, low=0)
                )

                v = numpyro.sample(
                    "v",
                    dist.TruncatedNormal(v_mean, v_scale, low=0)
                )

                lo = numpyro.sample(site.lo, dist.HalfNormal(lo_scale))

                gamma_scale_offset = numpyro.sample("gamma_scale_offset", dist.HalfCauchy(2.5))
                gamma_scale_slope = numpyro.sample("gamma_scale_slope", dist.HalfCauchy(2.5))

        """ Model """
        mean = numpyro.deterministic(
            site.mean,
            lo[feature1, participant] + \
            jnp.maximum(
                0,
                -1 + \
                (h[feature1, participant] + 1) / \
                jnp.power(
                    1 + \
                    (jnp.power(1 + h[feature1, participant], v[feature1, participant]) - 1) * \
                    jnp.exp(-b[feature1, participant] * (intensity - a[feature1, participant])),
                    1 / v[feature1, participant]
                )
            )
        )

        scale = numpyro.deterministic(
            "scale",
            gamma_scale_offset[feature1, participant] + \
            gamma_scale_slope[feature1, participant] * (1 / mean)
        )

        """ Observation """
        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(mean * scale, scale),
                obs=response_obs
            )

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference
        """
        response = df[RESPONSE].to_numpy().reshape(-1,)
        participant = df[PARTICIPANT].to_numpy().reshape(-1,)
        feature1 = df[FEATURES[1]].to_numpy().reshape(-1,)
        intensity = df[INTENSITY].to_numpy().reshape(-1,)

        # MCMC
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info(f"Running inference with {self.name} ...")
        mcmc.run(rng_key, intensity, participant, feature1, response)
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples

    def _get_threshold_estimates(
        self,
        combination: tuple,
        posterior_samples: dict,
        prob: float = .95
    ):
        threshold_posterior = posterior_samples[site.a][
            :, combination[1], combination[0]
        ]
        threshold = threshold_posterior.mean()
        hpdi_interval = hpdi(threshold_posterior, prob=prob)
        return threshold, threshold_posterior, hpdi_interval

    def predict(
        self,
        intensity: np.ndarray,
        combination: tuple,
        posterior_samples: Optional[dict] = None,
        num_samples: int = 100
    ):
        predictive = Predictive(model=self._model, num_samples=num_samples)
        if posterior_samples is not None:
            predictive = Predictive(model=self._model, posterior_samples=posterior_samples)

        participant = np.repeat([combination[0]], intensity.shape[0])
        feature1 = np.repeat([combination[1]], intensity.shape[0])

        predictions = predictive(
            self.rng_key,
            intensity=intensity,
            participant=participant,
            feature1=feature1
        )
        return predictions
