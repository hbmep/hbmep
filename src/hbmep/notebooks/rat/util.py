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
        "hbmep",
        "notebooks",
        "rat",
        "loghb",
        experiment.lower()[2:].replace('_', "")
    )
    toml_path = os.path.join(
        REPOS,
        "hbmep",
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
    colors = np.where(cmap_arr > .6, "k", "white")

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


def load_csmalar_data(data: pd.DataFrame):
    data = data.copy()
    # make sure columns channel1_segment and channel2_segment are correct
    ch1 = data.compound_position.apply(lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][:2])
    pd.testing.assert_series_equal(ch1, data.channel1_segment, check_names=False)
    ch2 = data.compound_position.apply(lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][:2])
    pd.testing.assert_series_equal(ch2, data.channel2_segment, check_names=False)
    # make sure channel2_segment is never nan
    assert not data.channel2_segment.isna().any()
    # make sure columns channel1_designation and channel2_designation are correct
    ch1_lat = data.compound_position.apply(lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][2:])
    pd.testing.assert_series_equal(ch1_lat, data.channel1_designation, check_names=False)
    ch2_lat = data.compound_position.apply(lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][2:])
    pd.testing.assert_series_equal(ch2_lat, data.channel2_designation, check_names=False)
    # make sure channel2_designation is never nan
    assert not data.channel2_designation.isna().any()

    # create the relevant feature columns
    data["segment"] = np.where(
        data.channel1_segment.isna(),
        "-" + data.channel2_segment,
        data.channel1_segment + "-" + data.channel2_segment
    )
    assert not data["segment"].isna().any()
    data["lat"] = np.where(
        data.channel1_designation.isna(),
        "-" + data.channel2_designation,
        data.channel1_designation + "-" + data.channel2_designation
    )
    assert not data["lat"].isna().any()
    # make sure the new feature columns are correct
    temp = (
        data[["segment", "lat"]]
        .apply(
            lambda x: x[0].split("-")[0] + x[1].split("-")[0] + "-" + x[0].split("-")[1] + x[1].split("-")[1],
            axis=1
        )
    )
    pd.testing.assert_series_equal(temp, data.compound_position, check_names=False)

    df = data.copy()
    # Remove contacts with size B-S
    remove_size = ["B-S"]
    idx = df.compound_size.isin(remove_size)
    df = df[~idx].reset_index(drop=True).copy()
    # Remove contacts with designation RM, R, RR
    remove_designation = ["RM", "R", "RR"]
    idx = df.channel1_designation.isin(remove_designation)
    df = df[~idx].reset_index(drop=True).copy()
    idx = df.channel2_designation.isin(remove_designation)
    df = df[~idx].reset_index(drop=True).copy()
    # Remove C7 segment
    remove_segments = ["C7"]
    idx = df.channel1_segment.isin(remove_segments)
    df = df[~idx].reset_index(drop=True).copy()
    idx = df.channel2_segment.isin(remove_segments)
    df = df[~idx].reset_index(drop=True).copy()
    # Remove bipolar contacts that connect between two different segments.
    # these were recorded by mistake during experiments and won't be analyzed
    idx = (df.channel1_segment == df.channel2_segment) | df.channel1_segment.isna()
    df = df[idx].reset_index(drop=True).copy()
    assert ((df.channel1_segment == df.channel2_segment) | df.channel1_segment.isna()).all()
    return df
