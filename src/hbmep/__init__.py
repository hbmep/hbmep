import logging
import multiprocessing
from importlib.metadata import version

import numpyro

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

__version__ = version("hbmep")
logger = logging.getLogger(__name__)

# cpu_count = multiprocessing.cpu_count()
# numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()
