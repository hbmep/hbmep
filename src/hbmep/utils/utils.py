import logging
from time import time
from functools import wraps

import numpy as np
from numpyro.diagnostics import hpdi

logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_taken = te - ts
        hours_taken = time_taken // (60 * 60)
        time_taken %= (60 * 60)
        minutes_taken = time_taken // 60
        time_taken %= 60
        seconds_taken = time_taken % 60
        if hours_taken:
            message = \
                f"func:{f.__name__} took: {hours_taken:0.0f} hr and " + \
                f"{minutes_taken:0.0f} min"
        elif minutes_taken:
            message = \
                f"func:{f.__name__} took: {minutes_taken:0.0f} min and " + \
                f"{seconds_taken:0.2f} sec"
        else:
            message = f"func:{f.__name__} took: {seconds_taken:0.2f} sec"
        logger.info(message)
        return result
    return wrap


def floor(x: float, base: float = 10):
    result = base * np.floor(x / base)
    if base > 0: return int(result)
    return result


def ceil(x: float, base: float = 10):
    result = base * np.ceil(x / base)
    if base > 0: return int(result)
    return result


def evaluate_posterior_mean(posterior_samples, prob: float = .95):
    posterior_mean = posterior_samples.mean(axis=0)
    return posterior_mean


def evaluate_hpdi_interval(posterior_samples, prob: float = .95):
    hpdi_interval = hpdi(posterior_samples, prob=prob)
    return hpdi_interval
