import os
import logging
import multiprocessing

import arviz as az
import jax
import numpyro

from hbmep.config import Config
from hbmep.model import Model

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()

logger = logging.getLogger(__name__)


def run_inference(config: Config):
    model = Model(config=config)

    """ Preprocess """
    df, encoder_dict = model.load()
    model.plot(df=df, encoder_dict=encoder_dict)

    """ Run inference """
    mcmc, posterior_samples = model.run_inference(df=df)

    """ Save inference data """
    model.save(mcmc=mcmc)

    """ Recruitment curves """
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)

    """ Posterior Predictive Check """
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)

    logger.info(f"Finished saving artefacts to {model.model.build_dir}")
    return
