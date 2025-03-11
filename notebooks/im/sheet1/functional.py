import jax
import jax.numpy as jnp

from hbmep.functional import *


def function(x, b3, b4, b1, b2):
    z = (x / b3) ** (-b4)
    z = b2 / (1 + z)
    z = b1 + z
    return z
