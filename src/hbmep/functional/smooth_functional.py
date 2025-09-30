import warnings

from hbmep.functional.util import (
    smooth_max,
    linear_transform,
    logistic_transform,
    get_threshold,
)

EPS = 1e-3


def rectified_logistic(x, a, b, g, h, v, eps=EPS):
    r"""
    Smooth approximation of the rectified-logistic function
    """
    warnings.warn(
        "Use hbmep.functional.rectified_logistic(..., eps=...) instead.",
        DeprecationWarning, stacklevel=2
    )
    z = logistic_transform(x, a, b, h, v)
    z = smooth_max(z, eps)
    return g + z


def rectified_linear(x, a, b, g, eps=EPS):
    r"""
    Smooth approximation of the rectified-linear function
    """
    warnings.warn(
        "Use hbmep.functional.rectified_linear(..., eps=...) instead.",
        DeprecationWarning, stacklevel=2
    )
    z = linear_transform(x, a, b)
    z = smooth_max(z, eps)
    return g + z


def rectified_logistic_inInflectionParam(x, a, b, g, h, v, eps=EPS):
    r"""
    Smooth approximation of rectified-logistic function in S50 parameterization
    """
    warnings.warn(
        "Use rectified_logistic_inflection_param(..., eps=...) instead.",
        DeprecationWarning, stacklevel=2
    )
    a = get_threshold(a, b, g, h, v)
    return rectified_logistic(x, a, b, g, h, v, eps)
