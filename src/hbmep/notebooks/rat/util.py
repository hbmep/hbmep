import os
import pickle
import logging 

import numpy as np
import pandas as pd
from hbmep.util import site

from hbmep.notebooks.constants import DATA, REPOS, REPORTS
from hbmep.notebooks.rat.constants import (
    circ as circ_constants,
    shie as shie_constants,
    smalar as smalar_constants
)
logger = logging.getLogger(__name__)


def get_paths(experiment): # example - get_paths("L_CIRC")
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
    # model.plot(df, encoder=encoder)
    mcmc, posterior = model.run(df=df, **kw)

    # Save
    output_path = os.path.join(model.build_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model.__dict__,), f)
    logger.info(f"Saved to {output_path}")

    if mcmc is not None:
        output_path = os.path.join(model.build_dir, "mcmc.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f)
        logger.info(f"Saved to {output_path}")

    predict(df, encoder, posterior, model, mcmc)
    return


def predict(df, encoder, posterior, model, mcmc):
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


def load_model(model_dir):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior, = pickle.load(f)

    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)

    mcmc = None
    try:
        src = os.path.join(model_dir, "mcmc.pkl")
        with open(src, "rb") as f:
            mcmc, = pickle.load(f)
    except FileNotFoundError:
        logger.info("mcmc.pkl not found. Attempting to read from model_dict.pkl")
    except ValueError as e:
        logger.info("Encountered ValueError, trace is below")
        logger.info(e)
    else:
        logger.info("Found mcmc.pkl")

    if mcmc is None:
        try:
            src = os.path.join(model_dir, "model_dict.pkl")
            with open(src, "rb") as f:
                mcmc, _ = pickle.load(f)
        except FileNotFoundError:
            logger.info("model_dict.pkl not found.")
        except ValueError as e:
            logger.info("Encountered ValueError, trace is below. Possibly issue with unpacking.")
            logger.info(e)
        else:
            logger.info("Found model_dict.pkl")

    return df, encoder, posterior, model, mcmc


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


def load_circ(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
    **kw
):
    _, _, DATA_PATH, _ = get_paths(circ_constants.EXPERIMENT)

    MAP = circ_constants.MAP
    DIAM = circ_constants.DIAM
    VERTICES = circ_constants.VERTICES
    RADII = circ_constants.RADII
    
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    df = log_transform_intensity(data, intensity)

    cats = df[features[1]].unique().tolist()
    mapping = {}
    for cat in cats:
        assert cat not in mapping
        l, r = cat.split("-")
        mapping[cat] = l[3:] + "-" + r[3:]
    assert mapping == MAP
    df[features[1]] = df[features[1]].replace(mapping)
    cats = set(df[features[1]].tolist())
    assert set(DIAM) <= cats
    assert set(VERTICES) <= cats
    assert set(RADII) <= cats
    df = df.copy()

    assert run_id in {"diam", "radii", "vertices", "all"}
    match run_id:
        case "diam": subset = DIAM
        case "radii": subset = RADII
        case "vertices": subset = VERTICES
        case "all": subset = DIAM + RADII + VERTICES
        case _: raise ValueError

    if set_reference:
        match run_id:
            case "diam" | "radii": reference = "-C"; subset += [reference]
            case "vertices": reference = "S-N"; subset += [reference]
            case "all": reference = "-C"
            case _: raise ValueError

    assert set(subset) <= set(df[features[1]].values.tolist())
    ind = df[features[1]].isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    if set_reference:
        df[features[1]] = df[features[1]].replace({reference: " " + reference})

    return df
    


def load_shie(
    *,
    intensity,
    features,
    run_id,
    **kw
):
    _, _, DATA_PATH, _ = get_paths(shie_constants.EXPERIMENT)

    POSITIONS_MAP = shie_constants.POSITIONS_MAP
    CHARGES_MAP = shie_constants.CHARGES_MAP
    WITH_GROUND = shie_constants.WITH_GROUND
    NO_GROUND = shie_constants.NO_GROUND

    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    df = log_transform_intensity(data, intensity)
    df[features[1]] = df[features[1]].replace(POSITIONS_MAP)
    df[features[2]] = df[features[2]].replace(CHARGES_MAP)

    assert run_id in {"ground", "no-ground", "all"}
    match run_id:
        case "ground": subset = WITH_GROUND
        case "no-ground": subset = NO_GROUND
        case "all": subset = WITH_GROUND + NO_GROUND
        case _: raise ValueError
    assert set(subset) <= set(df[features[1:]].apply(tuple, axis=1).values.tolist())
    ind = df[features[1:]].apply(tuple, axis=1).isin(subset)
    df = df[ind].reset_index(drop=True).copy()
    return df


def load_size(
    *,
    intensity,
    features,
    run_id,
    **kw
):
    DATA_PATH_FILTERED = smalar_constants.DATA_PATH_FILTERED
    NO_GROUND = smalar_constants.NO_GROUND
    GROUND = smalar_constants.GROUND
    GROUND_BIG = smalar_constants.GROUND_BIG
    GROUND_SMALL = smalar_constants.GROUND_SMALL
    NO_GROUND_BIG = smalar_constants.NO_GROUND_BIG
    NO_GROUND_SMALL = smalar_constants.NO_GROUND_SMALL

    # Load data
    src = DATA_PATH_FILTERED
    data = pd.read_csv(src)
    df = log_transform_intensity(data, intensity)
    
    assert run_id in {"ground", "no-ground", "all"}
    subset = []
    match run_id:
        case "ground": subset = GROUND
        case "no-ground": subset = NO_GROUND
        case "all": subset = (
            GROUND
            + NO_GROUND
            + GROUND_BIG
            + GROUND_SMALL
            + NO_GROUND_BIG
            + NO_GROUND_SMALL
        ); subset = list(set(subset))
        case _: raise ValueError
    assert len(set(subset)) == len(subset)
    cols = ["lat", "segment", "compound_size"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    df[features[-2]] = df[features[-2]].replace(
        {"-LM1": "-LM", "M-LM1": "M-LM"}
    )
    # df[model.features[-3]] = df[model.features[-3]].replace(
    #     {"C5-C5": "-C5", "C6-C6": "-C6"}
    # )
    return df


def load_lat(
    *,
    intensity,
    features,
    run_id,
    **kw
):
    DATA_PATH_FILTERED = smalar_constants.DATA_PATH_FILTERED
    GROUND_BIG = smalar_constants.GROUND_BIG
    GROUND_SMALL = smalar_constants.GROUND_SMALL
    NO_GROUND_BIG = smalar_constants.NO_GROUND_BIG
    NO_GROUND_SMALL = smalar_constants.NO_GROUND_SMALL

    # Load data
    src = DATA_PATH_FILTERED
    data = pd.read_csv(src)
    df = log_transform_intensity(data, intensity)
    
    assert run_id in {"small-ground", "big-ground", "small-no-ground", "big-no-ground"}
    subset = []
    match run_id:
        case "small-ground": subset = GROUND_SMALL
        case "big-ground": subset = GROUND_BIG
        case "small-no-ground": subset = NO_GROUND_SMALL
        case "big-no-ground": subset = NO_GROUND_BIG
        case _: raise ValueError
    assert len(set(subset)) == len(subset)
    cols = ["lat", "segment", "compound_size"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    df[features[-1]] = df[features[-1]].replace(
        {"-LM1": "-LM", "M-LM1": "M-LM"}
    )
    # df[model.features[-2]] = df[model.features[-3]].replace(
    #     {"C5-C5": "-C5", "C6-C6": "-C6"}
    # )
    return
