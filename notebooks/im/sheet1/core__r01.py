import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from jax import random, numpy as jnp
from hbmep.util import timing, setup_logging

from models import ImmunoModel
import functional as RF
from utils import Site as site
from constants import (
    DATA_PATH,
    BUILD_DIR,
)


@timing
def main(model):
    df = pd.read_csv(DATA_PATH)
    df, encoder = model.load(df, mask_non_positive=False)
    idx = df[model.features[0]].isin([0])
    df = df[idx].reset_index(drop=True).copy()

    mcmc, posterior = model.run(df)
    mcmc.print_summary()
    prediction_df = model.make_prediction_dataset(df)
    predictive = model.predict(prediction_df, posterior=posterior)
    posterior.keys()
    model.plot_curves(
        df, prediction_df=df, predictive=posterior, encoder=encoder
    )
    # model.plot_curves(
    #     df, prediction_df=prediction_df, predictive=predictive, encoder=encoder, prediction_prob=.95
    # )
    return



if __name__ == "__main__":
    model = ImmunoModel()
    # model._model = model.nhb_r01
    model._model = model.nhb_lognormal
    model.build_dir = os.path.join(
        BUILD_DIR, model._model.__name__
    )
    setup_logging(model.build_dir)
    main(model)
