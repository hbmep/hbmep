import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging, site

from models import HB
# from hbmep.notebooks.rat.util import run
from constants import (
    BUILD_DIR,
    TOML_PATH,
    DATA_PATH,
    MAP,
    DIAM,
    VERTICES,
    RADII
)

logger = logging.getLogger(__name__)

@timing
def run(df, model):
    df, encoder = model.load(df)
    svi_result, posterior = model.run_svi(df)
    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior)
    try:
        if site.a not in posterior.keys():
            posterior[site.a] = predictive[site.a]
    except: pass
    model.plot_curves(
        df=df,
        encoder=encoder,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
    )

    a_loc = posterior[site.a.loc]
    a_loc.shape
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.close("all")
    sns.kdeplot(a_loc)
    plt.show()

    model.__features

    return


@timing
def main(model):
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    data[model.intensity] = 1 + data[model.intensity]
    # idx = data[model.intensity] > 0
    # data = data[idx].reset_index(drop=True).copy()
    data[model.intensity] = np.log2(data[model.intensity])

    cats = data[model.features[1]].unique().tolist()
    mapping = {}
    for cat in cats:
        assert cat not in mapping
        l, r = cat.split("-")
        mapping[cat] = l[3:] + "-" + r[3:]
    assert mapping == MAP
    data[model.features[1]] = data[model.features[1]].replace(mapping)
    cats = set(data[model.features[1]].tolist())
    assert set(DIAM) <= cats
    assert set(VERTICES) <= cats
    assert set(RADII) <= cats
    df = data.copy()

    run_id = model.run_id
    assert run_id in {"diam", "radii", "vertices", "all"}
    match run_id:
        case "diam": subset = DIAM
        case "radii": subset = RADII
        case "vertices": subset = VERTICES
        case "all": subset = DIAM + RADII + VERTICES
        case _: raise ValueError
    assert set(subset) <= set(df[model.features[1]].values.tolist())
    ind = df[model.features[1]].isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    subset = ["amap01", "amap02"]
    idx = df[model.features[0]].isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    # model.response = model.response[:2]

    # model.features = [model.features]
    logger.info(f"*** run id: {run_id} ***")
    logger.info(f"*** model: {model._model.__name__} ***")
    run(df, model)
    return


if __name__ == "__main__":
    model = HB(toml_path=TOML_PATH)
    model.build_dir = os.path.join(BUILD_DIR, "svi", model.run_id, model._model.__name__)
    setup_logging(model.build_dir)
    main(model)
