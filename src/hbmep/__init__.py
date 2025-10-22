import os
import multiprocessing
cpu_count = multiprocessing.cpu_count()
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={cpu_count}")

import jax
jax.config.update("jax_enable_x64", True)

import numpyro
numpyro.enable_validation()

from importlib.metadata import version
__version__ = version("hbmep")

from hbmep.functional import functional, smooth_functional
from hbmep import invert, integrate
from hbmep.util import site
from hbmep.dataset import (
    load,
    fit_transform,
    inverse_transform,
    make_features,
    make_prediction_dataset,
)
from hbmep.plotter import plotter, plot
from hbmep.model import (
    BaseModel,
    NonHierarchicalBaseModel,
    StandardHB
)
from hbmep.infer import (
    get_regressors,
    get_response,
    get_dependencies,
    trace,
    run,
    predict,
)
