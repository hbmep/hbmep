import os
import pickle
import logging

# import numpy as np
from hbmep.util import setup_logging

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.util import load_model, predict, site

logger = logging.getLogger(__name__)


def main(model_dir):
    df, encoder, posterior, model, mcmc = load_model(model_dir)
    model.build_dir = model_dir
    setup_logging(os.path.join(model.build_dir, "predictive.log"))
    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_predictive(df, encoder=encoder, prediction_df=prediction_df, predictive=predictive)
    return


if __name__ == "__main__":
    model_dirs = [
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",

        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/ground/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",

        "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/no-ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/no-ground/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked"
    ]
    [main(model_dir) for model_dir in model_dirs]
