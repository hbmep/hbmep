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

from hb_mep.config import HBMepConfig
from hb_mep.models.baseline import Baseline
from hb_mep.models.utils import Site as site
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


class MixedEffects(Baseline):
    def __init__(self, config: HBMepConfig):
        super(MixedEffects, self).__init__(config=config)
        self.name = "Mixed_Effects"

        self.columns = [PARTICIPANT, FEATURES[1]]
        self.x = np.linspace(0, 15, 100)
        self.xpad = 2

    def _model(self, intensity, participant, feature1, response_obs=None):
        n_data = intensity.shape[0]
        n_participant = np.unique(participant).shape[0]
        n_feature1 = np.unique(feature1).shape[0]

        """ Hyperpriors """
        delta_mean = numpyro.sample(site.delta_mean, dist.Normal(0, 10))
        delta_scale = numpyro.sample(site.delta_scale, dist.HalfNormal(2.0))

        # Baseline threshold
        baseline_mean_global_mean = numpyro.sample(
            "baseline_mean_global_mean",
            dist.HalfNormal(2.0)
        )
        baseline_scale_global_scale = numpyro.sample(
            site.baseline_scale_global_scale,
            dist.HalfNormal(2.0)
        )

        baseline_scale = numpyro.sample(
            site.baseline_scale,
            dist.HalfNormal(baseline_scale_global_scale)
        )
        baseline_mean = numpyro.sample(
            site.baseline_mean,
            dist.TruncatedDistribution(dist.Normal(baseline_mean_global_mean, baseline_scale), low=0)
        )

        # Slope
        b_mean_global_scale = numpyro.sample(site.b_mean_global_scale, dist.HalfNormal(5.0))
        b_scale_global_scale = numpyro.sample(site.b_scale_global_scale, dist.HalfNormal(2.0))

        b_mean = numpyro.sample(site.b_mean, dist.HalfNormal(b_mean_global_scale))
        b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(b_scale_global_scale))

        # MEP at rest
        lo_scale_global_scale = numpyro.sample(site.lo_scale_global_scale, dist.HalfNormal(2.0))
        lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(lo_scale_global_scale))

        # # Saturation
        # g_shape = numpyro.sample(site.g_shape, dist.HalfNormal(20.0))

        with numpyro.plate("n_participant", n_participant, dim=-1):
            """ Priors """
            baseline = numpyro.sample(
                site.baseline,
                dist.TruncatedDistribution(dist.Normal(baseline_mean, baseline_scale), low=0)
            )
            delta = numpyro.sample(site.delta, dist.Normal(delta_mean, delta_scale))

            with numpyro.plate("n_feature1", n_feature1, dim=-2):
                # Threshold
                a = numpyro.deterministic(
                    site.a,
                    jnp.array([baseline, baseline + delta])
                )

                # Slope
                b = numpyro.sample(
                    site.b,
                    dist.TruncatedDistribution(dist.Normal(b_mean, b_scale), low=0)
                )

                # MEP at rest
                lo = numpyro.sample(site.lo, dist.HalfNormal(lo_scale))

                # # Saturation
                # g = numpyro.sample(site.g, dist.Beta(1, g_shape))

                # Noise
                noise_offset = numpyro.sample(
                    site.noise_offset,
                    dist.HalfCauchy(0.5)
                )
                noise_slope = numpyro.sample(
                    site.noise_slope,
                    dist.HalfCauchy(0.5)
                )

        # Model
        mean = numpyro.deterministic(
            site.mean,
            lo[feature1, participant] + \
            jax.nn.relu(b[feature1, participant] * (intensity - a[feature1, participant]))
        )

        sigma = numpyro.deterministic(
            "sigma",
            noise_offset[feature1, participant] + \
            noise_slope[feature1, participant] * mean
        )

        penalty = 5 * (jnp.fabs(baseline + delta) - (baseline + delta))
        numpyro.factor(site.penalty, -penalty)

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=response_obs)

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
