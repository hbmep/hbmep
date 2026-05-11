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

from hbmep.functional import functional, inverse, integrate
from hbmep.dataset import (
    process,
    fit_transform,
    inverse_transform,
    make_features,
    make_prediction_dataset,
)
from hbmep.plotter import plot
from hbmep.infer import (
    get_regressors,
    get_response,
    get_dependencies,
    trace,
    run,
    predict,
)
from hbmep.model import (
    BaseModel,
    NonHierarchicalBaseModel,
    StandardHB,
)
from hbmep.model.util import save, load
from hbmep import metrics
from hbmep.device import (
    use_gpu,
    execute_on_gpu,
)
from hbmep.util import (
    site,
    timing,
    enable_logging,
    _enable_fallback_logging,
    make_pdf,
)

_enable_fallback_logging()
