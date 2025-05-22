import jax
import jax.numpy as jnp

from hbmep.functional import *


def ro1(x, b3, b4, b1, b2):
    z = b1 + (b2 / (
        1 + ((x / b3) ** (-b4))
    ))
    return z
