import jax
from jax import numpy as jnp, lax

from hbmep.functional.util import (
    linear_transform,
    logistic_transform,
    smooth_max,
)


def rectified_logistic(x, a, b, g, h, v, eps: float = 0.0):
    r"""
    Rectified-logistic function.

    .. math::

        \mathcal{f}\left(x; a, b, g, h, v\right)
        \;=\; g \;+\; \max\left\{0, \;-v + \frac{h + v}{1 + \left(\frac{h}{v}\right)e^{-b\left(x-a\right)}} \right\}

    If ``eps > 0``, replace the outer :math:`\max(0,\cdot)` with a smooth
    surrogate :math:`\text{smoothmax}(\cdot,\varepsilon)` with bandwidth :math:`\varepsilon`, given by

    .. math::

        \text{smoothmax}(t; \varepsilon) \;=\;
        t \;+\; \frac{\varepsilon}{\ln 2}\cdot \ln\left(1 + e^{-(t\ln 2)/\varepsilon}\right)


    Parameters
    ----------
    x : Array
        Input.
    a : Array or float
        Threshold parameter.
    b : Array or float
        Controls growth rate.
    g : Array or float
        Lower offset (baseline).
    h : Array or float
        Vertical distance to upper asymptote from offset (upper asymptote is :math:`g+h`).
    v : Array or float
        Controls the location of inflection point (>0).
    eps : float, optional
        If > 0, use smooth maximum for the rectifier with bandwidth
        :math:`\varepsilon`; default 0.0 (hard ReLU).

    Returns
    -------
    Array
        :math:`f(x; a,b,g,h,v)`.

    Notes
    -----
    - With ``eps == 0`` the function is non-differentiable at points where
      :math:`x=a`; with ``eps > 0`` it is everywhere smooth.
    - This is equivalent to applying a rectifier to a shifted and scaled
      logistic body.
    """
    z = logistic_transform(x, a, b, h, v)
    eps = jnp.asarray(eps, dtype=z.dtype)
    z = lax.cond(
        eps > 0,
        lambda t: smooth_max(t[0], t[1]),
        lambda t: jax.nn.relu(t[0]),
        (z, eps),
    )
    return g + z


def logistic5(x, a, b, g, h, v):
    r"""
    Logistic-5 (5-parameter logistic, with asymmetry :math:`v>0`).

    .. math::

        f(x; a, b, g, h, v) \;=\;
        g \;+\; \frac{h}{\left\{1 + \left(2^v - 1\right)e^{-b\left(x-a\right)}\right\}^{\frac1{v}}}

    Parameters
    ----------
    x : Array or float
        Input.
    a : Array or float
        :math:`\text{S}_{50}` parameter.
    b : Array or float
        Controls growth rate.
    g : Array or float
        Lower asymptote (offset, baseline).
    h : Array or float
        Vertical distance to upper asymptote from offset (upper asymptote is :math:`g+h`).
    v : Array or float
        Asymmetry (>0). :math:`v=1` reduces to the 4-parameter logistic.

    Returns
    -------
    Array
        :math:`f(x; a,b,g,h,v)`.
    """
    z = linear_transform(x, a, b) - jnp.log(-1 + jnp.power(2, v))
    z = jax.nn.sigmoid(z)
    z = jnp.power(z, 1 / v)
    z = h * z
    return g + z


def logistic4(x, a, b, g, h):
    r"""
    Logistic-4 (4-parameter logistic).

    .. math::

        f(x; a, b, g, h)
        \;=\; g \;+\; \frac{h}{1 + e^{-b\left(x-a\right)}}

    Parameters
    ----------
    x : Array or float
        Input.
    a : Array or float
        :math:`\text{S}_{50}` parameter (inflection occurs at :math:`x=a`).
    b : Array or float
        Controls growth rate.
    g : Array or float
        Lower asymptote (offset, baseline).
    h : Array or float
        Vertical distance to upper asymptote from offset (upper asymptote is :math:`g+h`).

    Returns
    -------
    Array
        :math:`f(x; a,b,g,h)`.
    """
    z = linear_transform(x, a, b)
    z = jax.nn.sigmoid(z)
    z = h * z
    return g + z


def rectified_linear(x, a, b, g):
    r"""
    Rectified-linear function.

    .. math::
        
        f(x; a,b,g) \;=\; g \;+\; \max\left\{0,\; b\left(x - a\right)\right\}

    If ``eps > 0``, replace the outer :math:`\max(0,\cdot)` with a smooth
    surrogate, same as in :func:`hbmep.functional.rectified_logistic`.

    Parameters
    ----------
    x : Array or float
        Input.
    a : Array or float
        Threshold parameter.
    b : Array or float
        Slope of the linear part.
    g : Array or float
        Baseline offset.
    eps : float, optional
        If > 0, use smooth maximum for the rectifier with bandwidth
        :math:`\varepsilon`; default 0.0 (hard ReLU).

    Returns
    -------
    Array
        :math:`f(x; a,b,g)`.

    Notes
    -----
    - With ``eps == 0`` the function is non-differentiable at :math:`x=a`;
      with ``eps > 0`` it is everywhere smooth.
    """
    z = linear_transform(x, a, b)
    eps = jnp.asarray(eps, dtype=z.dtype)
    z = lax.cond(
        eps > 0,
        lambda t: smooth_max(t[0], t[1]),
        lambda t: jax.nn.relu(t[0]),
        (z, eps),
    )
    return g + z


# # TODO: add these
# def rectified_logistic_inInflectionParam(x, a, b, g, h, v):
#     """
#     Rectified-logistic function in inflection parameterization
#     """
#     a = get_threshold(a, b, g, h, v)
#     return rectified_logistic(x, a, b, g, h, v)


# def grad(fn, x, *args):
#     """ Compute the gradient of a function """
#     args = jnp.broadcast_arrays(x, *args)
#     grad_fn = jax.grad(fn)
#     for _ in range(x.ndim):
#         grad_fn = jax.vmap(grad_fn)
#     return grad_fn(*args)
