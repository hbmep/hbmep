import os
import pickle
import logging

import numpy as np
from hbmep.util import Site as site

from hbmep.notebooks.rat.model import nHB
from constants import HOME, BUILD_DIR, INFERENCE_FILE, MODEL_FILE

logger = logging.getLogger(__name__)


def main():
    src = "/home/vishu/reports/hbmep/notebooks/rat/lognhb/lshie/nHB__4000W_1000S_4C_1T_20D_95A_mixtureTrue/inference.pkl"
    with open(src, "rb") as f:
        df, encoder, posterior, _ = pickle.load(f)

    src = "/home/vishu/reports/hbmep/notebooks/rat/lognhb/lshie/nHB__4000W_1000S_4C_1T_20D_95A_mixtureTrue/model.pkl"
    with open(src, "rb") as f:
        model, = pickle.load(f)

    posterior.keys()
    for u, v in posterior.items():
        print(u, v.shape, np.isnan(v).sum())

    named_params = [site.a, site.b, site.L, site.ell, site.H]
    params = [posterior[named_param] for named_param in named_params]
    a, b, L, ell, H = params
    print(a.shape)

    model.features
    
    df_inverse = df.copy()
    df_inverse[model.features] = df_inverse[model.features].apply(lambda x: encoder[x.name].inverse_transform(x))
    combinations_inverse = df_inverse[model.features].apply(tuple, axis=1).unique().tolist()
    return
    


if __name__ == "__main__":
    main()