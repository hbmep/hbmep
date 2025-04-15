import os
import pickle

import numpy as np
import pandas as pd

from hbmep.model import BaseModel
from hbmep.util import site, setup_logging, timing

from hbmep.notebooks.rat.model import HB
from core__combined import CONFIG


# model_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked_mmax0/L_CIRC___L_SHIE___C_SMA_LAR/h_prior_0.1__conc1_1"
# model_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked_mmax1/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML/h_prior_0.1"
model_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked_mmax0/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML/h_prior_0.1__conc1_10/"
model_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML/"
base_model = BaseModel(config=CONFIG)

df = None
encoder = None
posterior = None

for respond_idx, response in enumerate(base_model.response):
    response_dir = os.path.join(model_dir, response)
    # if response == "LBiceps":
    #     print(f"Reading alt {response}...")
    #     response_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked_mmax2/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML/h_prior_0.1__conc1_10/LBiceps"
    src = os.path.join(response_dir, "inf.pkl")
    with open(src, "rb") as f:
        curr_df, curr_encoder, curr_posterior, = pickle.load(f)
    src = os.path.join(response_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)

    if df is None:
        df = curr_df.copy()
        encoder = curr_encoder
        posterior = curr_posterior
    else:
        pd.testing.assert_frame_equal(df, curr_df)
        for u, v in curr_encoder.items():
            assert v.classes_.tolist() == encoder[u].classes_.tolist()
        for u, v in posterior.items():
            posterior[u] = np.concatenate([v, curr_posterior[u]], axis=-1)
    print(f"Processed {response}")

model.response = base_model.response.copy()
model.build_dir = model_dir
for u, v in posterior.items(): print(u, v.shape)

named_params = [site.a, site.b, site.g, site.h, "h_fraction", "h_max", "h_max_fraction", "h_max_global"]
posterior = {u: posterior[u] for u in named_params if u in posterior.keys()}
for u, v in posterior.items(): print(u, v.shape)

num_features = df[model.features].max().to_numpy() + 1
mask_features = np.full((*num_features,), False)
_, features = model.get_regressors(df)
mask_features[*features.T] = True
mask_features = mask_features[None, ..., None]
print(mask_features.shape)


def body_mask(named_param):
    masked_arr = np.where(mask_features, posterior[named_param], np.nan)
    return masked_arr


def body_check_1(named_params):
    for u in named_params:
        try: v = posterior[u]
        except KeyError: print(f"Skipping {u}..."); continue
        mask_on = v[:, mask_features[0, ..., 0], :]
        mask_off = v[:, ~mask_features[0, ..., 0], :]
        bool_mask_on = np.isnan(mask_on).any()
        bool_mask_off = np.isnan(mask_off).all()
        assert (not bool_mask_on) and bool_mask_off
        print(f"{u} ok.")
    import inspect
    print(f"{inspect.currentframe().f_code.co_name} success.")
    return


named_params = [site.a, site.b, site.g, site.h, "h_fraction"]
posterior = {u: body_mask(u) if u in named_params else v for u, v in posterior.items()}
for u, v in posterior.items(): print(u, v.shape)
body_check_1(named_params)

output_path = os.path.join(model_dir, "combined_inf.pkl")
with open(output_path, "wb") as f:
    pickle.dump((df, encoder, posterior,), f)
print(f"Saved to {output_path}")

output_path = os.path.join(model_dir, "combined_model.pkl")
with open(output_path, "wb") as f:
    pickle.dump((model,), f)
print(f"Saved to {output_path}")

output_path = os.path.join(model_dir, "combined_mask.pkl")
with open(output_path, "wb") as f:
    pickle.dump((num_features, mask_features,), f)
print(f"Saved to {output_path}")

print(0)
