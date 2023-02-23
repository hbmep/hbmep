import os
import logging
from pathlib import Path
from operator import itemgetter

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import graphviz

from hb_mep.config import HBMepConfig
from hb_mep.utils.constants import (
    REPORTS_DIR,
    INTENSITY,
    MEP_SIZE,
    PARTICIPANT,
    SEGMENT
)

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

logger = logging.getLogger(__name__)


class Baseline():
    def __init__(self, config: HBMepConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

        self.random_state = 0

    def model(self, intensity, participant, segment, mep_size_obs=None):
        a_level_scale_global_scale = numpyro.sample('a_global_scale', dist.HalfNormal(2.0))
        b_level_scale_global_scale = numpyro.sample('b_global_scale', dist.HalfNormal(2.0))

        a_level_mean_global_scale = numpyro.sample('a_level_mean_global_scale', dist.HalfNormal(5.0))
        b_level_mean_global_scale = numpyro.sample('b_level_mean_global_scale', dist.HalfNormal(5.0))

        sigma_offset_level_scale_global_scale = \
            numpyro.sample('sigma_offset_level_scale_global_scale', dist.HalfCauchy(5.0))
        sigma_slope_level_scale_global_scale = \
            numpyro.sample('sigma_slope_level_scale_global_scale', dist.HalfCauchy(5.0))

        n_participants = np.unique(participant).shape[0]
        n_levels = np.unique(segment).shape[0]

        with numpyro.plate("n_levels", n_levels, dim=-2):
            a_level_mean = numpyro.sample("a_level_mean", dist.HalfNormal(a_level_mean_global_scale))
            b_level_mean = numpyro.sample("b_level_mean", dist.HalfNormal(b_level_mean_global_scale))

            a_level_scale = numpyro.sample("a_level_scale", dist.HalfNormal(a_level_scale_global_scale))
            b_level_scale = numpyro.sample("b_level_scale", dist.HalfNormal(b_level_scale_global_scale))

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

            with numpyro.plate("n_participants", n_participants, dim=-1):
                a = numpyro.sample("a", dist.Normal(a_level_mean, a_level_scale))
                b = numpyro.sample("b", dist.Normal(b_level_mean, b_level_scale))

                sigma_offset = numpyro.sample('sigma_offset', dist.HalfCauchy(sigma_offset_level_scale))
                sigma_slope = numpyro.sample('sigma_slope', dist.HalfCauchy(sigma_slope_level_scale))

        mean = jax.nn.relu(b[segment, participant] * (intensity - a[segment, participant]))
        sigma = sigma_offset[segment, participant] + sigma_slope[segment, participant] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=mep_size_obs)

    def render(
        self,
        data_dict: dict
        ) -> graphviz.graphs.Digraph:
        """
        Render NumPyro model and save resultant graph.

        Args:
            model (model): NumPyro model for rendering.
            data_dict (dict): Data dictionary containing model parameters for rendering.
            filename (Optional[Path], optional): Target destination for saving rendered graph. Defaults to None.

        Returns:
            graphviz.graphs.Digraph: Rendered graph.
        """
        logger.info('Rendering model ...')
        # Retrieve data from data dictionary for rendering model
        intensity, mep_size, participant, segment  = \
            itemgetter(INTENSITY, MEP_SIZE, PARTICIPANT, SEGMENT)(data_dict)
        return numpyro.render_model(
            self.model,
            model_args=(intensity, participant, segment, mep_size),
            filename=os.path.join(self.reports_path, self.config.RENDER_FNAME)
        )

    def sample(self, data_dict: dict) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference

        Args:
            data_dict (dict): Data dictionary containing input and observations

        Returns:
            tuple[numpyro.infer.mcmc.MCMC, dict]: MCMC inference results and posterior samples.
        """
        # Retrieve data from data dictionary
        intensity, mep_size, participant, segment  = \
            itemgetter(INTENSITY, MEP_SIZE, PARTICIPANT, SEGMENT)(data_dict)

        # MCMC
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info('Running inference ...')
        mcmc.run(rng_key, intensity, participant, segment, mep_size)
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples
