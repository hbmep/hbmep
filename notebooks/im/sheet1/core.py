import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from jax import random, numpy as jnp
from hbmep.util import site, timing, setup_logging

from models import ImmunoModel
from constants import (
    DATA_PATH,
    BUILD_DIR,
)


def main(model):
    df = pd.read_csv(DATA_PATH)
    df[model.features[0]] = df[model.features[0]].astype(str).copy()
    idx = df[model.intensity] > 0
    df = df[idx].reset_index(drop=True).copy()
    df[model.intensity] = np.log2(df[model.intensity])
    df, encoder = model.load(df)
    # model.plot(df, encoder=encoder)
    mcmc, posterior = model.run(df)
    prediction_df = model.make_prediction_dataset(df)
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df, prediction_df=prediction_df, predictive=predictive, encoder=encoder, prediction_prob=.95
    )
    return



if __name__ == "__main__":
    model = ImmunoModel()
    model._model = model.nhb_logistic4
    model.build_dir = os.path.join(
        BUILD_DIR, model._model.__name__
    )
    setup_logging(model.build_dir)
    main(model)
