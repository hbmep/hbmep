import os
import pickle
import logging

from hbmep.util import setup_logging

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.util import load_model, predict, site

logger = logging.getLogger(__name__)


def main(model_dir):
    setup_logging(os.path.join(model_dir, "predict.log"))
    logger.info(f"Predicting for {model_dir}...")
    df, encoder, posterior, model, mcmc = load_model(model_dir)
    model.build_dir = model_dir
    predict(df, encoder, posterior, model, mcmc)
    return


if __name__ == "__main__":
    model_dirs = [

        # # lognormal models

        # circ
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/radii/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/vertices/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",

        # shie
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/ground/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/no-ground/4000w_4000s_4c_4t_15d_95a_fm/ln_hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/all/4000w_4000s_4c_4t_15d_95a_tm/ln_hb_mvn_rl_nov_masked",

        ## C_SMA_LAR lat models (XIO: fatal IO error 22 (Invalid argument) on X server)
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/big-no-ground/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/small-ground/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/small-no-ground/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/big-ground/hb_mvn_rl_nov_masked/",

    ]
    [main(model_dir) for model_dir in model_dirs]
