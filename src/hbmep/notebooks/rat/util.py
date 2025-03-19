import os
import pickle
import logging 

import numpy as np
import pandas as pd
from hbmep.util import site

from hbmep.notebooks.constants import DATA, REPOS, REPORTS
logger = logging.getLogger(__name__)


def get_paths(experiment):
    build_dir = os.path.join(
        REPORTS,
        "notebooks",
        "rat",
        "loghb",
        experiment.lower().split('_')[1]
    )
    toml_path = os.path.join(
        REPOS,
        "configs",
        "rat",
        f"{experiment}.toml"
    )
    data_path = os.path.join(
        DATA,
        "rat",
        experiment,
        "data.csv"
    )
    mep_matrix_path = os.path.join(
        DATA,
        "rat",
        experiment,
        "mat.npy"
    )
    return build_dir, toml_path, data_path, mep_matrix_path


def log_transform_intensity(df: pd.DataFrame, intensity: str):
    data = df.copy()
    intensities = sorted(data[intensity].unique().tolist())
    min_intensity = intensities[0]
    assert min_intensity >= 0
    if min_intensity > 0: pass
    else:
        logger.info(f"Minimum intensity is {min_intensity}. Handling this before taking log2...")
        replace_zero_with = 2 ** -1
        assert replace_zero_with < intensities[1]
        logger.info(f"Replacing {min_intensity} with {replace_zero_with}")
        data[intensity] = data[intensity].replace({min_intensity: replace_zero_with})
        intensities = sorted(data[intensity].unique().tolist())[:5]
        logger.info(f"New minimum intensities: {intensities}")
    data[intensity] = np.log2(data[intensity])
    return data


def run(data, model, encoder=None, **kw):
    # Run
    if encoder is None:
        df, encoder = model.load(df=data)
    else:
        df = data.copy()
    logger.info(f"df.shape {df.shape}")
    mcmc, posterior = model.run(df=df, **kw)

    # Save
    output_path = os.path.join(model.build_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)

    output_path = os.path.join(model.build_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)

    output_path = os.path.join(model.build_dir, "model_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((mcmc, model.__dict__,), f)

    # Predictions
    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df=df,
        encoder=encoder,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
    )

    if site.outlier_prob in posterior.keys():
        posterior.pop(site.outlier_prob)
    summary_df = model.summary(posterior)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")
    logger.info(f"Finished running {model.name}")
    try:
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        logger.info(f"No. of divergences {divergences}")
        num_steps = mcmc.get_extra_fields()["num_steps"]
        tree_depth = np.floor(np.log2(num_steps)).astype(int)
        logger.info(f"Tree depth statistics:")
        logger.info(f"Min: {tree_depth.min()}")
        logger.info(f"Max: {tree_depth.max()}")
        logger.info(f"Mean: {tree_depth.mean()}")
    except: pass
    logger.info(f"Saved results to {model.build_dir}")
    return


def get_subname(model):
    return (
        f'{model.mcmc_params["num_warmup"]}w'
        f'_{model.mcmc_params["num_samples"]}s'
        f'_{model.mcmc_params["num_chains"]}c'
        f'_{model.mcmc_params["thinning"]}t'
        f'_{model.nuts_params["max_tree_depth"][0]}d'
        f'_{model.nuts_params["target_accept_prob"] * 100:.0f}a'
        f'_{"t" if model.use_mixture else "f"}m'
    )


def mask_upper(arr):
    n = arr.shape[0]
    arr[np.triu_indices(n)] = np.nan
    return arr


def annotate_heatmap(ax, cmap_arr, arr, l, r, star=False, star_arr=None, **kw):
    n = arr.shape[0]
    colors = np.where(cmap_arr > .8, "k", "white")

    for y in range(n):
        for x in range(n):
            if x >= y: continue
            text = f"{arr[y, x]}"
            if star:
                pvalue = star_arr[y, x]
                if pvalue < 0.001: text += "***"
                elif pvalue < 0.01: text += "**"
                elif pvalue < 0.05: text += "*"
            # ax.text(x + l, y + r, (y, x), **kw, color=colors[y, x])
            ax.text(x + l, y + r, text, **kw, color=colors[y, x])
