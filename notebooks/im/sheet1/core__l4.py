import os
import pickle

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


@timing
def main(model):
    data = pd.read_csv(DATA_PATH)
    df = load(data)
    idx = df[model.intensity] > 0
    df = df[idx].reset_index(drop=True).copy()
    df[model.intensity] = np.log2(df[model.intensity])

    df, encoder = model.load(df)
    # model.plot(df, encoder=encoder)

    mcmc, posterior = model.run(df)
    prediction_df = model.make_prediction_dataset(df, num_points=500)
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df, prediction_df=prediction_df, predictive=predictive, encoder=encoder, prediction_prob=.95
    )

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
    model._model = model.hb1_l4
    model.build_dir = os.path.join(
        BUILD_DIR, model._model.__name__
    )
    setup_logging(model.build_dir)
    main(model)
