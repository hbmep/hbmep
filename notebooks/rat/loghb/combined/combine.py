import os
import pickle

import numpy as np
from jax import numpy as jnp
import pandas as pd
from hbmep.util import site

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.util import load_model
from core__hb import CONFIG

RESPONSE = CONFIG["variables"]["response"]


def mask(named_param, posterior, mask_features):
    v = posterior[named_param]
    if v.shape == mask_features.shape:
        v_masked = np.where(mask_features, np.nan, v)
    else:
        print(f"Can't mask {named_param} bc. shape mismatch")
        v_masked = v
    return v_masked


def check1(named_params, posterior, mask_features):
    for u in named_params:
        try: 
            v = posterior[u]
        except KeyError: 
            print(f"Can't check {u} bc. it doesn't exist.")
            continue
        else:
            if v.shape == mask_features.shape:
                mask_true = v[mask_features]
                mask_false = v[~mask_features]
                bool_all_nan = np.isnan(mask_true).all()
                bool_any_nan = np.isnan(mask_false).any()
                assert (not bool_any_nan) and bool_all_nan
                print(f"{u} ok.")
            else:
                print(f"Can't check {u} bc. shape mismatch")
    import inspect
    print(f"{inspect.currentframe().f_code.co_name} success.")
    return


def process(model_dir):
    df = None
    encoder = None
    posterior = None

    for respond_idx, response in enumerate(RESPONSE):
        response_dir = os.path.join(model_dir, response)
        (
            curr_df, curr_encoder, curr_posterior, model, _
        ) = load_model(response_dir)
        if site.outlier_prob in curr_posterior:
            curr_posterior[site.outlier_prob] *= 0
        curr_posterior = {
            u: v for u, v in curr_posterior.items()
            if v.ndim in [1, 4]
        }
        curr_posterior = {
            u: v[..., None] if v.ndim == 1 else v
            for u, v in curr_posterior.items()
        }
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

    model.response = RESPONSE
    model.build_dir = model_dir
    for u, v in posterior.items(): print(u, v.shape)
    # posterior = {
    #     u: np.mean(v, axis=-1) if v.ndim == 2 else v
    #     for u, v in posterior.items()
    # }
    # for u, v in posterior.items(): print(u, v.shape)

    num_features = df[model.features].max().to_numpy() + 1
    num_samples = (
        (model.mcmc_params["num_samples"] * model.mcmc_params["num_chains"])
        // model.mcmc_params["thinning"]
    )
    mask_features = np.full(
        (num_samples, *num_features, model.num_response), True
    )
    _, features = model.get_regressors(df)
    mask_features[:, *features.T, :] = False
    t1 = np.sum(~mask_features) / (num_samples * model.num_response)
    t2 = df[model.features].apply(tuple, axis=1).nunique()
    assert t1 == t2

    posterior = {
        u: mask(u, posterior, mask_features)
        for u in posterior.keys()
    }
    named_params = [site.a, site.b, site.g, site.h, site.v, "h_max"]
    check1(named_params, posterior, mask_features)

    if "h_max" not in posterior.keys():
        print(f"Adding h_max using emperical max...")
        h = posterior[site.h].copy()
        h_max = np.nanmax(h, axis=-2, keepdims=True)
        posterior["h_max"] = h_max

    g = posterior[site.g].copy()
    h_max = posterior["h_max"].copy()
    response = df[model.response].to_numpy().copy()
    response = response[None, ...] - g[:, *features.T]
    response /= jnp.array(h_max)[:, *features.T]; response = np.array(response)
    response += g[:, *features.T]
    response = np.mean(response, axis=0)
    assert response.min() > 0
    norm_response = [f"norm_{r}" for r in model.response]
    df[norm_response] = response
    model.norm_response = norm_response
    assert not np.any(posterior[site.h] > posterior["h_max"])

    output_path = os.path.join(model_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)
    print(f"Saved to {output_path}")
    output_path = os.path.join(model_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    print(f"Saved to {output_path}")
    output_path = os.path.join(model_dir, "mask.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((num_features, mask_features,), f)
    print(f"Saved to {output_path}")

    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] *= 0
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df=df,
        encoder=encoder,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
    )
    return


def main():
    model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/combined_data/L_CIRC___L_SHIE___C_SMA_LAR/4000w_4000s_4c_4t_15d_95a_tm/hb_rl_masked"
    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/combined_data/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML/4000w_4000s_4c_4t_15d_95a_tm/hb_rl_masked"
    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_rl_masked_hmaxPooled/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML"
    process(model_dir)


if __name__ == "__main__":
    main()
