import os
import logging
from time import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
FORMAT =  "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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


def setup_logging(output, *, level=logging.INFO, format=FORMAT):
    root, ext = os.path.splitext(output)
    if not ext: output_file = os.path.join(output, "logs.log")
    else: output_file = output

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        format=format,
        level=level,
        handlers=[
            logging.FileHandler(output_file, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {output_file}")


def abstractvariables(*args):
    """Decorator to enforce required subclass attributes."""
    def decorator(cls):
        original_init = cls.__init__

        def wrapped_init(self, *init_args, **init_kwargs):
            # Call the original constructor
            original_init(self, *init_args, **init_kwargs)

            # Check that required attributes are defined in the subclass
            for attr, message in args:
                if not hasattr(self, attr):
                    raise NotImplementedError(f"Descendants must set variable `{attr}`. {message}")

        cls.__init__ = wrapped_init
        return cls

    return decorator


def floor(x: float, base: float = 10):
    return base * np.floor(x / base)


def ceil(x: float, base: float = 10):
    return base * np.ceil(x / base)


def invert_combination(
    combination: tuple[int],
    columns: list[str],
    encoder_dict: dict[str, LabelEncoder]
) -> tuple:
    return tuple(
        encoder_dict[column].inverse_transform(np.array([value]))[0]
        for (column, value) in zip(columns, combination)
    )


def generate_colors(n: int):
    return plt.cm.rainbow(np.linspace(0, 1, n))


class Site:
    # Priors
    a = "a"
    b = "b"
    v = "v"
    L = "L"
    ell = "ℓ"
    H = "H"

    c1 = "c₁"
    c2 = "c₂"

    # Deterministic
    mu = "µ"
    alpha = "α"
    beta = "β"

    # Plates
    num_features = [f"num_feature{i}" for i in range(10)]
    num_response = "num_response"
    num_data = "num_data"

    # Observation
    obs = "obs"

    # Mixture
    outlier_prob = "p_outlier"
    outlier_scale = "σ_outlier"
    q = "q"
    bg_scale = "σ_bg"

    # S50
    s50 = "S50"

    # Outlier classifier
    p = "p"

