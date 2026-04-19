import gc
import json
import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from numpyro.infer import MCMC
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

POSTERIOR_DIR = "posterior"
POSTERIOR_KEYS_PKL = "keys.pkl"
POSTERIOR_META_JSON = "meta.json"
DATA_FILE = "data.pkl"
MODEL_STATE_FILE = "model_state.pkl"
MCMC_FILE = "mcmc.pkl"


def save(
    *,
    model_state: dict | None = None,
    df: pd.DataFrame | None = None,
    posterior: dict[str, np.ndarray] | None = None,
    encoder: dict[str, LabelEncoder] | None = None,
    mcmc: MCMC | None = None,
    output_dir: str | None = None,
) -> None:
    """
    Save hbMEP outputs to a directory.

    Parameters
    ----------
    model_state
        Output of `model.state_dict()`.
    df
        Dataframe to save.
    posterior
        Posterior samples as a dict of arrays.
    encoder
        Feature encoders returned by `model.load(...)`.
    mcmc
        NumPyro MCMC object.
    output_dir
        Destination directory.

    Notes
    -----
    - Any argument may be None, in which case it is skipped.
    - Posterior arrays are saved separately as `.npy` files under
      `{output_dir}/posterior/`.
    """
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    os.makedirs(output_dir, exist_ok=True)

    if model_state is not None:
        output_path = os.path.join(output_dir, MODEL_STATE_FILE)
        with open(output_path, "wb") as f:
            pickle.dump((model_state,), f, protocol=pickle.HIGHEST_PROTOCOL)

    if df is not None or encoder is not None:
        output_path = os.path.join(output_dir, DATA_FILE)
        with open(output_path, "wb") as f:
            pickle.dump((df, encoder), f, protocol=pickle.HIGHEST_PROTOCOL)

    if mcmc is not None:
        output_path = os.path.join(output_dir, MCMC_FILE)
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f, protocol=pickle.HIGHEST_PROTOCOL)

    if posterior is not None:
        posterior_dir = os.path.join(output_dir, POSTERIOR_DIR)
        os.makedirs(posterior_dir, exist_ok=True)

        keys = list(posterior.keys())
        output_path = os.path.join(posterior_dir, POSTERIOR_KEYS_PKL)
        with open(output_path, "wb") as f:
            pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)

        items_meta = []
        num_draws = None
        for i, key in enumerate(keys):
            arr = np.asarray(posterior[key])

            if arr.ndim == 0:
                raise ValueError(
                    f"Posterior array for key '{key}' must have at least 1 dimension"
                )

            if num_draws is None:
                num_draws = int(arr.shape[0])
            elif arr.shape[0] != num_draws:
                raise ValueError(
                    "All posterior arrays must share the same first dimension "
                    f"(num draws). Got {num_draws} and {arr.shape[0]} for key '{key}'."
                )

            fname = f"arr_{i}.npy"
            np.save(os.path.join(posterior_dir, fname), arr)
            items_meta.append(
                {
                    "index": i,
                    "key": key,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "file": fname,
                }
            )

        meta = {
            "num_draws": 0 if num_draws is None else num_draws,
            "num_items": len(items_meta),
            "items": items_meta,
        }
        output_path = os.path.join(posterior_dir, POSTERIOR_META_JSON)
        with open(output_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    logger.info(f"Saved to {output_dir}")
    gc.collect()


def load(
    *,
    model_dir: str,
    memmap_draw: int | None = None,
    data_file: str = DATA_FILE,
    model_state_file: str = MODEL_STATE_FILE,
    mcmc_file: str = MCMC_FILE,
) -> tuple[
    dict | None,
    pd.DataFrame | None,
    dict[str, np.ndarray],
    dict[str, LabelEncoder] | None,
    MCMC | None,
]:
    """
    Load outputs saved with `save(...)`.

    Parameters
    ----------
    model_dir
        Directory containing saved outputs.
    data_file
        Pickle file containing `(df, encoder)`.
    model_state_file
        Pickle file containing `(model_state,)`.
    mcmc_file
        Pickle file containing `(mcmc,)`.
    memmap_draw
        If None, load full posterior arrays into RAM.
        If an integer, memory-map each posterior array and return only that
        draw slice with shape `(1, ...)`. In this case, `mcmc` is not loaded.

    Returns
    -------
    model_state, df, posterior, encoder, mcmc
    """
    model_state = None
    df, encoder = None, None
    posterior: dict[str, np.ndarray] = {}
    mcmc = None

    src = os.path.join(model_dir, model_state_file)
    try:
        with open(src, "rb") as f:
            model_state, = pickle.load(f)
    except FileNotFoundError:
        logger.info(f"{model_state_file} not found in {model_dir}")

    src = os.path.join(model_dir, data_file)
    try:
        with open(src, "rb") as f:
            df, encoder = pickle.load(f)
            if memmap_draw is not None and df is not None:
                df = df.iloc[[0]].copy()
    except FileNotFoundError:
        logger.info(f"{data_file} not found in {model_dir}")

    if memmap_draw is None:
        src = os.path.join(model_dir, mcmc_file)
        try:
            with open(src, "rb") as f:
                mcmc, = pickle.load(f)
        except FileNotFoundError:
            logger.info(f"{mcmc_file} not found in {model_dir}")
        except Exception as e:
            logger.info(f"MCMC load info: {e}")

    posterior_dir = os.path.join(model_dir, POSTERIOR_DIR)
    if os.path.isdir(posterior_dir):
        src = os.path.join(posterior_dir, POSTERIOR_KEYS_PKL)
        with open(src, "rb") as f:
            keys_list = pickle.load(f)

        src = os.path.join(posterior_dir, POSTERIOR_META_JSON)
        with open(src, "r") as f:
            meta = json.load(f)

        num_draws = int(meta.get("num_draws", 0))
        if memmap_draw is not None:
            if not (0 <= memmap_draw < num_draws):
                raise IndexError(
                    f"memmap_draw {memmap_draw} out of range [0, {num_draws - 1}]"
                )

        items = sorted(meta["items"], key=lambda d: d["index"])
        if len(items) != len(keys_list):
            raise ValueError("Posterior keys and items length mismatch")

        for i, key in enumerate(keys_list):
            fname = items[i]["file"]
            src = os.path.join(posterior_dir, fname)
            if memmap_draw is None:
                posterior[key] = np.load(src, mmap_mode=None)
            else:
                mm = np.load(src, mmap_mode="r")
                posterior[key] = mm[memmap_draw:memmap_draw + 1]

    return model_state, df, posterior, encoder, mcmc
