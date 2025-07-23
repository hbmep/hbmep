import os
import pickle

import numpy as np
from scipy import stats

from hbmep.util import site
from hbmep.notebooks.rat.model import HB, Estimation


def load_circ(model_dir):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)

    num_features = df[model.features].max().to_numpy() + 1
    subjects = sorted(df[model.features[0]].unique())
    subjects_inv = encoder[model.features[0]].inverse_transform(subjects)
    subjects = list(zip(subjects, subjects_inv))

    positions = sorted(df[model.features[1]].unique())
    positions_inv = encoder[model.features[1]].inverse_transform(positions)
    positions = list(zip(positions, positions_inv))

    return (
        df,
		encoder,
		model,
		posterior,
		subjects,
		positions,
		num_features
    )


def load_shie(model_dir):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)

    num_features = df[model.features].max().to_numpy() + 1
    subjects = sorted(df[model.features[0]].unique())
    subjects_inv = encoder[model.features[0]].inverse_transform(subjects)
    subjects = list(zip(subjects, subjects_inv))

    positions = sorted(df[model.features[1]].unique())
    positions_inv = encoder[model.features[1]].inverse_transform(positions)
    positions = list(zip(positions, positions_inv))

    try:
        charges = sorted(df[model.features[2]].unique())
        charges_inv = encoder[model.features[2]].inverse_transform(charges)
        charges = list(zip(charges, charges_inv))
    except IndexError:
        charges = None

    return (
        df,
		encoder,
		model,
		posterior,
		subjects,
		positions,
		charges,
		num_features,
    )


def load_smalar(model_dir, estimation=False):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)

    subjects = sorted(df[model.features[0]].unique())
    subjects_inv = encoder[model.features[0]].inverse_transform(subjects)
    subjects = list(zip(subjects, subjects_inv))

    feature_pos_idx, feature_deg_idx, feature_size_idx = 1, 2, 3
    if estimation:
        feature_pos_idx, feature_deg_idx = 2, 1
        if "size" in model.run_id:
            feature_pos_idx, feature_deg_idx = 3, 2
            feature_size_idx = 1

    positions = sorted(df[model.features[feature_pos_idx]].unique())
    positions_inv = encoder[model.features[feature_pos_idx]].inverse_transform(positions)
    positions = list(zip(positions, positions_inv))

    degrees = sorted(df[model.features[feature_deg_idx]].unique())
    degrees_inv = encoder[model.features[feature_deg_idx]].inverse_transform(degrees)
    degrees = list(zip(degrees, degrees_inv))

    try: size_feature = model.features[feature_size_idx]
    except IndexError as e: 
        print(f"Encoutered {e}. Possbily size feature is missing.")
        sizes = None
    else:
        sizes = sorted(df[size_feature].unique())
        sizes_inv = encoder[size_feature].inverse_transform(sizes)
        sizes = list(zip(sizes, sizes_inv))

    num_features, mask_features = None, None
    if not estimation:
        named_params = [site.a, site.b, site.g, site.h, site.v]
        posterior = {u: posterior[u] for u in named_params if u in posterior.keys()}
        # for u, v in posterior.items(): print(u, v.shape)

        num_features = df[model.features].max().to_numpy() + 1
        mask_features = np.full((*num_features,), False)
        _, features = model.get_regressors(df)
        mask_features[*features.T] = True
        mask_features = mask_features[None, ..., None]


        def body_mask(named_param):
            masked_arr = np.where(mask_features, posterior[named_param], np.nan)
            return masked_arr


        def body_check_1(named_params):
            for u in named_params:
                try: v = posterior[u]
                except KeyError: continue
                mask_on = v[:, mask_features[0, ..., 0], :]
                mask_off = v[:, ~mask_features[0, ..., 0], :]
                bool_mask_on = np.isnan(mask_on).any()
                bool_mask_off = np.isnan(mask_off).all()
                assert (not bool_mask_on) and bool_mask_off
                # print(f"{u} ok.")
            import inspect
            print(f"{inspect.currentframe().f_code.co_name} success.")
            return


        named_params = [site.a, site.b, site.g, site.h, site.v]
        posterior = {u: body_mask(u) if u in named_params else v for u, v in posterior.items()}
        # for u, v in posterior.items(): print(u, v.shape)
        body_check_1(named_params)

    return (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features
    )


# def cap_response(*, model, df, y_unnorm, func, func_params):
#     y = y_unnorm.copy()
#     num_features = df[model.features].max().to_numpy() + 1

#     # Get max at the highest tested intensity
#     xcap = np.full((*num_features, model.num_response), np.nan)
#     xcap.shape

#     gdf = df.groupby(model.features, as_index=False)[model.intensity].max()
#     features, max_intensity = gdf[model.features].values, gdf[model.intensity].values
#     xcap[*features.T] = max_intensity[:, None]
#     xcap.shape

#     # Sanity check
#     for combination in features:
#         idx = gdf[model.features].apply(tuple, axis=1).isin([tuple(combination)])
#         assert idx.sum() == 1
#         temp_intensity = gdf[idx].reset_index(drop=True)[model.intensity].values.item()
#         assert np.all(xcap[*combination] == temp_intensity)

#     ycap = np.array(func(
#         xcap[None, None, ...], *func_params
#     ))
#     y = np.where(y < ycap, y, ycap)
#     return y


# def evaluate_response(func, x, **params):
#     print("Evaluating y_response...")
#     # for u, v in params.items(): print(u, v.shape)
#     # print(f"x {x.shape}")
#     y = np.array(func(x, **params))
#     # print(f"y_response.shape {y.shape}")

#     # assert np.all(np.isnan(y).any(axis=0) == np.isnan(y).all(axis=0))
#     # assert np.all(np.isnan(y).any(axis=1) == np.isnan(y).all(axis=1))
#     # assert np.all(np.isnan(y).any(axis=-1) == np.isnan(y).all(axis=-1))
#     # num_nans = np.isnan(y[0, 0, ..., 0]).sum()
#     # print(f"y_unnorm num_nans: {num_nans}")
#     # print(f"y_unnorm isnan ok.")

#     # assert np.all(np.isnan(y).any(axis=0) == np.isnan(y).all(axis=0))
#     # assert np.all(np.isnan(y).any(axis=1) == np.isnan(y).all(axis=1))
#     # assert np.all(np.isnan(y).any(axis=-1) == np.isnan(y).all(axis=-1))
#     # assert num_nans == np.isnan(y[0, 0, ..., 0]).sum()
#     # print(f"y_norm isnan ok.")
#     return y


# def evaluate_entropy(*, x, y):
    
#     # y (num_x_points, num_samples, f0, f1, f2, ..., fn, num_response)
#     p = np.sum(y, axis=-1, keepdims=True)
#     with ignore_warnings:
#         p_ = np.where(p, y / p, 1 / y.shape[-1])
#         p = np.where(np.isnan(y), np.nan, p_)
#         print(f"p.shape {p.shape}")

#     with ignore_warnings:
#         plogp_ = np.where(p, p * np.log(p), 0)
#         plogp = np.where(np.isnan(y), np.nan, plogp_)
#         print(f"plogp.shape {plogp.shape}")

#     entropy = 1 + (np.sum(plogp, axis=-1) / np.log(y.shape[-1]))
#     # entropy = 1 - (stats.entropy(p, axis=-1) / np.log(y.shape[-1]))
#     print(f"entropy.shape {entropy.shape}")
#     auc = np.trapz(entropy, x[..., 0], axis=0)
#     print(f"auc.shape {auc.shape}")
#     return p, plogp, entropy, auc


# def evaluate_entropy(
#     *, x, y_norm, num_nans
# ):
#     y = y_norm.copy()
#     p = np.sum(y, axis=-1, keepdims=True)
#     assert np.isnan(p[0, ...]).sum() == num_nans

#     with ignore_warnings:
#         p_ = np.where(p, y / p, 1 / y.shape[-1])
#         p = np.where(np.isnan(y), np.nan, p_)
#         print(p.shape)
#         assert np.isnan(p[0, ..., 0]).sum() == num_nans

#     with ignore_warnings:
#         plogp_ = np.where(p, p * np.log(p), 0)
#         plogp = np.where(np.isnan(y), np.nan, plogp_)
#         print(plogp.shape)
#         assert np.isnan(plogp[0, ..., 0]).sum() == num_nans

#     entropy_ = 1 + (np.sum(plogp, axis=-1) / np.log(y.shape[-1]))
#     entropy = 1 - (stats.entropy(p, axis=-1) / np.log(y.shape[-1]))
#     np.testing.assert_almost_equal(entropy, entropy_)
#     print(f"entropy.shape {entropy.shape}")
#     assert np.isnan(entropy[0, ...]).sum() == num_nans
#     assert np.all(np.isnan(entropy).any(axis=0) == np.isnan(entropy).all(axis=0))

#     auc = np.trapz(y=entropy, x=x, axis=0)
#     print(f"auc.shape {auc.shape}")
#     assert np.isnan(auc).sum() == num_nans

#     return (
#         p,
#         plogp,
#         entropy,
#         auc
#     )
