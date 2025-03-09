import logging
import multiprocessing
import importlib.metadata

import numpyro

from hbmep.dataset import (
    fit_transform,
    load,
    inverse_transform,
    make_prediction_dataset,
)
from hbmep.plotter import plotter
from hbmep.infer import (
    get_regressors,
    get_response,
    get_dependencies,
    run,
    predict,
)

__version__ = importlib.metadata.version("hbmep")
cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
