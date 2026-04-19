import jax.numpy as jnp


def rectified_logistic(x, a, b, g, h, v):
    r"""
    Closed-form integral of the rectified-logistic function.

    .. math::
        F(x) = \int_0^x f(t; a,b,g,h,v)\, dt

    where :math:`f` is the rectified-logistic function.

    Notes
    -----
    The integral between two points :math:`x_1` and :math:`x_2` is
    :math:`F(x_2) - F(x_1)`.
    """
    def body_integrate(x):
        z = b * (x - a) - jnp.log(h) + jnp.log(v)
        z = jnp.log1p(jnp.exp(z))
        z *= (h + v) / b
        z += (g - v) * x
        return z

    z = body_integrate(jnp.maximum(x, a)) - body_integrate(a)
    z += g * jnp.minimum(x, a)
    return z


def rectified_logistic_around_threshold(x, a, b, g, h, v):
    r"""
    Closed form integral of rectified-logistic function
    from a to (a + x), where x > 0.
    Must satisfy np.nanmin(x) > 0.
    """
    z = (h / v) + jnp.exp(b * x)
    z /= 1 + (h / v)
    z = jnp.log(z) * ((h + v) / b)
    z += (g - v) * x
    return z
