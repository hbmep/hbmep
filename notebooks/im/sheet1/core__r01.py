import os
import pickle
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from jax import random, numpy as jnp
from hbmep.util import timing, setup_logging

from models import ImmunoModel
import functional as RF
from utils import Site as site, load
from constants import (
    DATA_PATH,
    BUILD_DIR,
)

logger = logging.getLogger(__name__)


@timing
def main(model):
    data = pd.read_csv(DATA_PATH)
    df = load(data)
    idx = df[model.intensity] > 0
    df = df[idx].reset_index(drop=True).copy()
    df[model.intensity] = np.log2(df[model.intensity])

    df, encoder = model.load(df)
    # model.plot(df, encoder=encoder)

    mcmc, posterior = model.run(df, extra_fields=["num_steps"])
    prediction_df = model.make_prediction_dataset(df, num_points=500)
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df, prediction_df=prediction_df, predictive=predictive, encoder=encoder, prediction_prob=.95
    )

    try:
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        logger.info(f"No. of divergences {divergences}")
        num_steps = mcmc.get_extra_fields()["num_steps"]
        tree_depth = np.floor(np.log2(num_steps)).astype(int)
        logger.info(f"Tree depth statistics:")
        logger.info(f"Min: {tree_depth.min()}")
        logger.info(f"Max: {tree_depth.max()}")
        logger.info(f"Mean: {tree_depth.mean()}")
    except: pass

    output_path = os.path.join(model.build_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior), f) 

    output_path = os.path.join(model.build_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f) 

    return



if __name__ == "__main__":
    model = ImmunoModel()
    # model._model = model.nhb_l4
    # model._model = model.hb1_l4
    model._model = model.nhb_ln
    model.build_dir = os.path.join(
        BUILD_DIR, model._model.__name__
    )
    setup_logging(model.build_dir)
    main(model)
