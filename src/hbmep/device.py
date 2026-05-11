from functools import wraps

import jax

DEV_CPU = jax.devices("cpu")[0]
try:
    gpu = jax.devices("gpu")[0]
except:
    gpu = DEV_CPU
DEV_GPU = gpu


def use_gpu():
    """
    Context manager to temporarily use GPU as default JAX device.
    """
    return jax.default_device(DEV_GPU)


def execute_on_gpu(func):
    """
    Inspired by YigitElma (https://github.com/jax-ml/jax/discussions/23079)
    Decorator to set default device to GPU for a function.

    Parameters
    ----------
    func : callable
        Function to decorate

    Returns
    -------
    wrapper : callable
        Decorated function that will run always on GPU even if the default 
        backend is set to CPU.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with jax.default_device(DEV_GPU):
            return func(*args, **kwargs)
    return wrapper
