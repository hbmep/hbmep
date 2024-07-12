import multiprocessing
import logging

# import jax
import numpyro

# PLATFORM = "cpu"
# jax.config.update("jax_platforms", PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
