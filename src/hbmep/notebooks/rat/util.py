import os
import pickle
import warnings
import logging 

from jax import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from hbmep.util import site

from hbmep.notebooks.constants import DATA, REPOS, REPORTS
from hbmep.notebooks.rat.constants import (
    circ as circ_constants,
    shie as shie_constants,
    smalar as smalar_constants,
    rcml as rcml_constants
)

SEPARATOR = "___"
COMBINATION_CDF = "combination_cdf"
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


def log_transform_intensity(df: pd.DataFrame, intensity: str, convert=True):
    # return df
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
    # model.plot(df, encoder=encoder); return
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


def load_model(
    model_dir,
    inference_file="inf.pkl",
    model_file="model.pkl",
    mcmc_file="mcmc.pkl",
):
    src = os.path.join(model_dir, inference_file)
    with open(src, "rb") as f:
        df, encoder, posterior, = pickle.load(f)

    src = os.path.join(model_dir, model_file)
    with open(src, "rb") as f:
        model, = pickle.load(f)

    mcmc = None
    try:
        src = os.path.join(model_dir, mcmc_file)
        with open(src, "rb") as f:
            mcmc, = pickle.load(f)
    except FileNotFoundError:
        logger.info(f"{mcmc_file} not found. Attempting to read from model_dict.pkl")
    except ValueError as e:
        logger.info("Encountered ValueError, trace is below")
        logger.info(e)
    else:
        logger.info(f"Found {model_file}")

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


def annotate_heatmap(ax, cmap_arr, arr, l, r, star=False, star_arr=None, fontsize=None, **kw):
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
            ax.text(x + l, y + r, text, **kw, color=colors[y, x], size=fontsize)


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
    set_reference=False,
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

    if set_reference:
        reference = ('-C', 'Biphasic')
        match run_id:
            case "no-ground": subset += [reference]
            case "ground" | "all": pass
            case _: raise ValueError

    assert set(subset) <= set(df[features[1:]].apply(tuple, axis=1).values.tolist())
    ind = df[features[1:]].apply(tuple, axis=1).isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    if set_reference:
        ind = df[features[1:]].apply(tuple, axis=1).isin([reference])
        assert df.loc[ind, features[1]].nunique() == 1
        assert df.loc[ind, features[1]].unique()[0] == "-C"
        df.loc[ind, features[1]] = " -C"

    return df


def load_lat(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
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
    
    assert run_id in {"lat-small-ground", "lat-big-ground", "lat-small-no-ground", "lat-big-no-ground"}
    subset = []
    match run_id:
        case "lat-small-ground": subset = GROUND_SMALL
        case "lat-big-ground": subset = GROUND_BIG
        case "lat-small-no-ground": subset = NO_GROUND_SMALL
        case "lat-big-no-ground": subset = NO_GROUND_BIG
        case _: raise ValueError

    if set_reference:
        reference = "-M"
        match run_id:
            case "lat-small-ground": pass
            case "lat-big-ground": pass
            case "lat-small-no-ground": subset += [('-M', '-C5', 'S'), ('-M', '-C6', 'S')]
            case "lat-big-no-ground": subset += [('-M', '-C5', 'B'), ('-M', '-C6', 'B')]
            case _: raise ValueError

    assert len(set(subset)) == len(subset)
    cols = ["lat", "segment", "compound_size"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    df[features[-1]] = df[features[-1]].replace(
        {"-LM1": "-LM", "M-LM1": "M-LM"}
    )

    if set_reference:
        if "no-ground" in run_id:
            df["segment"] = df["segment"].apply(lambda x: f"-{x.split('-')[-1]}")
        df["lat"] = df["lat"].replace({reference: " " + reference})

    # if set_reference:
    #     t = df.groupby(cols, as_index=False)[features[0]].agg(lambda x: x.nunique())
    #     keys = t[cols].apply(tuple, axis=1); values = t[features[0]]
    #     key, values = zip(*sorted(zip(keys, values)))
    #     print(list(zip(keys, values)))
    return df


def load_size(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
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
    
    assert run_id in {"size-ground", "size-no-ground", "all"}
    if run_id == "all": assert not set_reference
    subset = []
    match run_id:
        case "size-ground": subset = GROUND
        case "size-no-ground": subset = NO_GROUND
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

    if set_reference:
        df["compound_size"] = df["compound_size"].replace({"S": " S"})

    # if set_reference:
    #     t = df.groupby(cols, as_index=False)[features[0]].agg(lambda x: x.nunique())
    #     keys = t[cols].apply(tuple, axis=1); values = t[features[0]]
    #     key, values = zip(*sorted(zip(keys, values)))
    #     print(list(zip(keys, values)))

    return df


def load_rcml_data(data: pd.DataFrame):
    data = data.copy()
    ch1 = data.compound_position.apply(lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][:2])
    pd.testing.assert_series_equal(ch1, data.channel1_segment, check_names=False)
    ch2 = data.compound_position.apply(lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][:2])
    pd.testing.assert_series_equal(ch2, data.channel2_segment, check_names=False)
    # make sure channel2_segment is never nan
    assert not data.channel2_segment.isna().any()
    # make sure columns channel1_designation and channel2_designation are correct
    ch1_lat = data.compound_position.apply(lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][2:])
    pd.testing.assert_series_equal(ch1_lat, data.channel1_laterality, check_names=False)
    ch2_lat = data.compound_position.apply(lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][2:])
    pd.testing.assert_series_equal(ch2_lat, data.channel2_laterality, check_names=False)
    # make sure channel2_designation is never nan
    assert not data.channel2_laterality.isna().any()

    # create the relevant feature columns
    data["segment"] = np.where(
        data.channel1_segment.isna(),
        "-" + data.channel2_segment,
        data.channel1_segment + "-" + data.channel2_segment
    )
    assert not data["segment"].isna().any()
    data["lat"] = np.where(
        data.channel1_laterality.isna(),
        "-" + data.channel2_laterality,
        data.channel1_laterality + "-" + data.channel2_laterality
    )
    assert not data["lat"].isna().any()
    # make sure the new feature columns are correct
    temp = (
        data[["segment", "lat"]]
        .apply(
            lambda x: (
                x[0].split("-")[0] + x[1].split("-")[0] + "-" +
                x[0].split("-")[1] + x[1].split("-")[1]
            ),
            axis=1
        )
    )
    pd.testing.assert_series_equal(
        temp, data.compound_position, check_names=False
    )
    return data


def load_rcml(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
    **kw
):
    GROUND = rcml_constants.GROUND
    ROSTRAL_CAUDAL = rcml_constants.ROSTRAL_CAUDAL
    MIDLINE_LATERAL = rcml_constants.MIDLINE_LATERAL
    ROOT_ALIGNED = rcml_constants.ROOT_ALIGNED

    _, _, DATA_PATH, _ = get_paths(rcml_constants.EXPERIMENT)
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    data = load_rcml_data(data)
    df = log_transform_intensity(data, intensity)
    assert run_id in {"all", "ground", "rostral-caudal"}

    subset = []
    match run_id:
        case "ground": subset = GROUND
        case "rostral-caudal": subset = ROSTRAL_CAUDAL
        case "all": subset = (
            GROUND + ROSTRAL_CAUDAL + MIDLINE_LATERAL + ROOT_ALIGNED
        )
        case _: raise ValueError
    assert len(set(subset)) == len(subset)
    cols = ["segment", "lat"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    return df


def make_test(diff, mask=True, correction=False, use_nonparametric=False):
    with np.errstate(invalid="ignore"):
        if use_nonparametric:
            test = stats.wilcoxon(diff, axis=0, nan_policy="omit")
        else:
            test = stats.ttest_1samp(
                diff, axis=0, nan_policy="omit", popmean=0
            )
    pvalue = test.pvalue
    if mask: pvalue = mask_upper(pvalue)

    if correction:
        if mask:
            idx = np.tril_indices_from(pvalue, k=-1)
            corrected = stats.false_discovery_control(pvalue[idx])
            pvalue[idx] = corrected
        else:
            pvalue = stats.false_discovery_control(pvalue)

    _test = stats.ttest_1samp(diff, popmean=0, axis=0, nan_policy="omit")
    deg = _test.df
    me = np.nanmean(diff, axis=0)
    return pvalue, deg, me,


def make_plot(pvalue, deg, me, labels, figsize=None, ax=None, fontsize=None):
    num_labels = len(labels)
    if ax is None:
        if figsize is None: figsize = (1.5 * num_labels, .8 * num_labels)
        fig, axes = plt.subplots(
            1, 1, constrained_layout=True, squeeze=False, figsize=figsize
        )
        ax = axes[0, 0]
    else: fig = None; axes = None
    sns.heatmap(
        pvalue, annot=False, ax=ax, cbar=False, vmin=0, vmax=1,
        xticklabels=labels, yticklabels=labels,
    )
    # Annotate
    pvalue_annot_kws = {"ha": 'center', "va": 'center'}
    annotate_heatmap(
        ax, pvalue,  np.round(pvalue, 3), 0.5, 0.5, star=True,
        star_arr=pvalue, **pvalue_annot_kws, fontsize=fontsize
    )
    deg_annot_kws = {"ha": 'left', "va": 'bottom'}
    annotate_heatmap(
        ax, pvalue, 1 + deg, 0, 1, **deg_annot_kws, fontsize=fontsize
    )
    statistic_annot_kws = {"ha": 'center', "va": 'top'}
    annotate_heatmap(
        ax, pvalue, np.round(me, 3), 0.5, 0, **statistic_annot_kws,
        fontsize=fontsize
    )
    ax.set_xticklabels(labels=labels, rotation=25, ha="right", size="x-small")
    ax.set_yticklabels(labels=labels, rotation=0, size="xx-small")
    return fig, axes


def make_compare(
    diff, positions, correction=False, figsize=None, ax=None,
    fontsize=None, use_nonparametric=False
):
    _, labels = zip(*positions)
    pvalue, deg, me, = make_test(
        diff, correction=correction, use_nonparametric=use_nonparametric
    )
    fig, axes = make_plot(
        pvalue, deg, me, labels, figsize=figsize, ax=ax, fontsize=fontsize
    )
    return pvalue, deg, me, fig, axes


def make_compare3p(
    measure,
	positions,
    sort=True,
	correction=False,
	negate=False,
	fig=None,
	palette="viridis",
    consistent_colors=False,
	skip_heatmap=False,
    lineplot=False,
    use_nonparametric=False,
):
    idx, positions = zip(*positions)
    measure = measure[..., idx].copy()                              # (S, P, C)


    def body_pairwise(measure):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            measure_mean = np.nanmean(measure, axis=0)              # (P, C)
            diff = measure[..., None] - measure[..., None, :]       # (S, P, C, C)
            diff = np.nanmean(diff, axis=0)                         # (P, C, C)
            diff_mean = np.nanmean(diff, axis=0)                    # (C, C)
            diff_err = stats.sem(diff, axis=0, nan_policy="omit")   # (C, C)
        return measure_mean, diff, diff_mean, diff_err


    # Determine order and reference
    measure_mean, diff, diff_mean, diff_err = body_pairwise(measure)
    t = diff_mean < 0 if negate else diff_mean > 0
    t = t.sum(axis=-1)
    t = [(i, u, v) for i, (u, v) in enumerate(zip(positions, t))]
    if sort: t = sorted(t, key=lambda x: (x[-1], x[0]))
    t = [(u, v) for u, v, _ in t]
    idx, positions = zip(*t); positions = list(zip(range(len(positions)), positions))
    measure = measure[..., idx]
    measure_mean, diff, diff_mean, diff_err = body_pairwise(measure)
    subjects = [f"amap0{i + 1}" for i in range(measure_mean.shape[0])]

    # Set colors
    colors = sns.color_palette(palette=palette, n_colors=len(positions))
    t = [u for _, u in positions]
    if consistent_colors: t = sorted(t)
    colors = dict(zip(t, colors))

    # Plot
    if fig is None:
        nr, nc = 1, 3
        fig, axes = plt.subplots(
            *(nr, nc), figsize=(5 * nc, 3.2 * nr), squeeze=False, constrained_layout=True
        )
    else: fig, axes = fig

    df = pd.DataFrame(measure_mean, index=subjects, columns=positions)
    df = df.reset_index().melt(id_vars='index', var_name='position', value_name='value')
    df = df.rename(columns={'index': 'subject'}).sort_values(by="position")
    df_mean = df.groupby("position")["value"].mean().reset_index()
    # df_mean = df.groupby("position", observed=True)["value"].median().reset_index()
    df_mean = df_mean.sort_values(by="position")
    df.position = df.position.apply(lambda x: x[1])
    df_mean.position = df_mean.position.apply(lambda x: x[1])

    ax = axes[0, 0]; ax.clear()
    if lineplot:
        sns.lineplot(
            data=df,
			x='position',
			y='value',
			hue='subject',
			palette=["grey"] * len(subjects),
			ax=ax,
			legend=False,
			alpha=.4
        )
        sns.scatterplot(
            data=df,
			x="position",
			y="value",
			hue="position",
			palette=colors,
			zorder=10,
			ax=ax,
			edgecolor="w"
        )
        sns.scatterplot(
            data=df_mean,
            x="position",
            y="value",
            hue="position",
            palette=colors,
            s=80,
            zorder=20,
            ax=ax,
            legend=False,
            marker="^",
            facecolor="grey",
            edgecolor="w"
        )
    else:
        sns.lineplot(
            data=df,
			x="subject",
			y="value",
			hue="position",
			palette=colors,
			ax=ax,
			marker="o"
        )
    ax.get_legend().set_title("")
    ax.tick_params(axis="x", rotation=45, labelsize="xx-small")

    ax = axes[0, 1]; ax.clear()
    for pos_idx, pos_inv in positions:
        if negate:
            xme = diff_mean[-1, :]
            xerr = diff_err[..., -1, :]
        else:
            xme = diff_mean[:, -1]
            xerr = diff_err[..., :, -1]
        xme = xme[pos_idx]
        xerr = xerr[pos_idx]
        ax.errorbar(
            x=xme,
            xerr=xerr,
            y=pos_inv,
            fmt="o",
            ecolor=colors[pos_inv],
            color=colors[pos_inv],
        )
    ax.vlines(
        xme,
        linestyle="-",
        color=colors[positions[-1][1]],
        ymax=(len(positions) - 1),
        ymin=0
    )
    ax.tick_params(axis="x", rotation=25, labelsize="x-small")
    ax.tick_params(axis="y", rotation=0, labelsize="x-small")

    if not skip_heatmap:
        ax = axes[0, 2]; ax.clear()
        *_, = make_compare(
            (-diff) if negate else diff,
            positions,
            ax=ax,
            fontsize="xx-small",
            correction=correction,
            use_nonparametric=use_nonparametric,
        )
        ax.text(x=.9, y=.9, s=f"correction:{correction}", transform=ax.transAxes, ha='right', va="top")

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            ax.set_xlabel(""); ax.set_ylabel("")
            sides = ["top", "right"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()

    ax = axes[0, 0]
    ax.legend(bbox_to_anchor=(-.2, 1), loc="upper right", reverse=True, fontsize="x-small")
    # return (fig, axes), positions, diff_mean, diff_err, colors, negate
    return (fig, axes), positions, measure_mean, diff, diff_mean, diff_err, negate,


def make_compare3p_bar(
    measure,
	positions,
    negate=False,
	correction=False,
	fig=None,
	palette="viridis",
    consistent_colors=False,
	skip_heatmap=False,
    lineplot=False,
):
    idx, positions = zip(*positions)
    measure = measure[..., idx].copy()  # (S, P, C)
    # assert negate is False
    if negate: measure *= -1


    def body_pairwise(measure):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diff = np.nanmean(measure, axis=0)  # (P, C)
            diff_mean = np.nanmean(diff, axis=0)    # (C,)
            diff_err = stats.sem(diff, axis=0, nan_policy="omit")   # (C,)
        return diff, diff_mean, diff_err


    _, diff_mean, diff_err = body_pairwise(measure)
    t = [(i, u, v) for i, (u, v) in enumerate(zip(positions, diff_mean))]
    t = sorted(t, key=lambda x: (x[-1], x[0]))
    t = [(u, v) for u, v, _ in t]

    idx, positions = zip(*t); positions = list(zip(range(len(positions)), positions))
    measure = measure[..., idx]
    diff, diff_mean, diff_err = body_pairwise(measure)
    subjects = [f"amap0{i + 1}" for i in range(diff.shape[0])]

    # Set colors
    colors = sns.color_palette(palette=palette, n_colors=len(positions))
    t = [u for _, u in positions]
    if consistent_colors: t = sorted(t)
    colors = dict(zip(t, colors))

    # Plot
    if fig is None:
        nr, nc = 1, 2
        fig, axes = plt.subplots(
            *(nr, nc), figsize=(5 * nc, 3.2 * nr), squeeze=False, constrained_layout=True
        )
    else: fig, axes = fig

    df = pd.DataFrame(diff, index=subjects, columns=positions)
    df = df.reset_index().melt(
        id_vars='index', var_name='position', value_name='value'
    )
    df = df.rename(columns={'index': 'subject'})
    df = df.sort_values(by="position")
    df_mean = df.groupby("position")["value"].mean().reset_index()
    df_mean = df_mean.sort_values(by="position")
    df.position = df.position.apply(lambda x: x[1])
    df_mean.position = df_mean.position.apply(lambda x: x[1])

    ax = axes[0, 0]; ax.clear()
    if lineplot:
        sns.lineplot(
            data=df, x='position', y='value', hue='subject', palette=["grey"] * len(subjects), ax=ax, legend=False, alpha=.4
        )
        sns.scatterplot(
            data=df, x="position", y="value", hue="position", palette=colors, zorder=10, ax=ax, edgecolor="w"
        )
        sns.scatterplot(
            data=df_mean,
            x="position",
            y="value",
            hue="position",
            palette=colors,
            s=80,
            zorder=20,
            ax=ax,
            legend=False,
            marker="^",
            facecolor="grey",
            edgecolor="w"
        )
    else:
        sns.lineplot(
            data=df, x="subject", y="value", hue="position", palette=colors, ax=ax, marker="o"
        )
    ax.get_legend().set_title("")
    ax.tick_params(axis="x", rotation=45, labelsize="xx-small")

    ax = axes[0, 1]; ax.clear()
    for pos_idx, pos_inv in positions:
        ax.bar(
            x=pos_inv,
            height=diff_mean[pos_idx],
            color=colors[pos_inv],
        )
        ax.errorbar(
            x=pos_inv,
            y=diff_mean[pos_idx],
            yerr=diff_err[pos_idx],
            fmt='none',
            ecolor="k",
            capsize=5,
        )
    ax.tick_params(axis="x", rotation=25, labelsize="x-small")
    ax.tick_params(axis="y", rotation=0, labelsize="x-small")

    pvalue, deg, me, = make_test(diff, mask=False, correction=correction)
    pvalue = np.round(pvalue, 3)
    deg = 1 + deg

    xticklabels = [(pos_inv, pvalue[pos_idx], deg[pos_idx]) for pos_idx, pos_inv in positions]
    xticklabels = [(u, f"\np={v}, ", f"N={t}") for u, v, t in xticklabels]
    xticklabels = ["".join(u) for u in xticklabels]
    ax.set_xticks(range(4)); ax.set_xticklabels(xticklabels)
    ax.text(x=.1, y=.9, s=f"correction:{correction}", transform=ax.transAxes, ha='left', va="top")

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            ax.set_xlabel(""); ax.set_ylabel("")
            sides = ["top", "right"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()

    ax = axes[0, 0]
    ax.legend(bbox_to_anchor=(-.2, 1), loc="upper right", reverse=True)
    return (fig, axes), positions, diff_mean, diff_err, colors, negate,


def make_pdf(figs, output_path):
    print("Making pdf...")
    with PdfPages(output_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight') 
    print(f"Saved to {output_path}")
    return


def make_dump(args: tuple, output_path=None):
    with open(output_path, "wb") as f:
        pickle.dump(args, f)
    print(f"Dumped to {output_path}")
    return


def load_dump(src):
    with open(src, "rb") as f:
        args = pickle.load(f)
    return args


def _adjust_brightness(rgb_in, val):
    hsv = rgb_to_hsv(rgb_in)
    hsv[2] = val  # Adjust the brightness (value component in HSV)
    rgb_out = hsv_to_rgb(hsv)
    return rgb_out


def _get_cmap_muscles_alt():
    vec_muscle = np.array(["Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB", "ADM", "TA", "EDB", "AH", "FDI", "auc_target"])
    cmap_mus_dark = np.array([
        _adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.5),  # trapz
        np.array([1, 133, 113]) / 255,  # delt
        np.array([166, 97, 26]) / 255,  # biceps
        np.array([44, 123, 182]) / 255,  # triceps
        np.array([52, 0, 102]) / 255,  # ecr
        _adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.3),  # fcr
        np.array([208, 28, 139]) / 255,  # apb
        np.array([77, 172, 38]) / 255,  # adm
        np.array([215, 25, 28]) / 255,  # ta
        np.array([123, 50, 148]) / 255,  # edb
        _adjust_brightness(np.array([153, 79, 0]) / 256, 0.4),  # ah
        np.array([231, 226, 61]) / 255,  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])
    cmap_mus_light = np.array([
        _adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.8),  # trapz
        np.array([128, 205, 193]) / 255,  # delt
        np.array([223, 194, 125]) / 255,  # biceps
        np.array([171, 217, 233]) / 255,  # triceps
        _adjust_brightness(np.array([200, 40, 0]) / 255, 0.6),  # ecr
        _adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.6),  # fcr
        np.array([241, 182, 218]) / 255,  # apb
        np.array([184, 225, 134]) / 255,  # adm
        np.array([253, 174, 97]) / 255,  # ta
        np.array([194, 165, 207]) / 255,  # edb
        _adjust_brightness(np.array([153, 79, 0]) / 256, 0.6),  # ah
        _adjust_brightness(np.array([23, 54, 124]) / 256, 0.6),  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])
    # Create a DataFrame to hold muscle names and corresponding colors
    T_color = pd.DataFrame({
        'muscle': vec_muscle,
        'cmap_mus_light': [tuple(c) for c in cmap_mus_light],
        'cmap_mus_dark': [tuple(c) for c in cmap_mus_dark],
    })
    # Convert RGB to hex
    T_color['cmap_mus_light_hex'] = T_color['cmap_mus_light'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    T_color['cmap_mus_dark_hex'] = T_color['cmap_mus_dark'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    return cmap_mus_dark, cmap_mus_light, vec_muscle, T_color


def get_response_colors(response: list[str]):
    cmap_mus_dark, cmap_mus_light, vec_muscle, T_color = _get_cmap_muscles_alt()
    cmap_dict = dict(zip(vec_muscle, cmap_mus_dark))
    colors = []
    for response in response: colors.append(cmap_dict[response[1:]])
    return colors


def compare_less_than(key, left, right, n_iters=50):
    left = left.copy()
    right = right.copy()
    prob = []
    for _ in range(n_iters):
        key, subkey = random.split(key)
        left_shuffled = np.array(random.permutation(subkey, left))
        key, subkey = random.split(key)
        right_shuffled = np.array(random.permutation(subkey, right))
        prob.append((left_shuffled < right_shuffled).mean())
    prob = np.mean(np.array(prob))
    return key, prob


def arg_mode(a, axis, argmax=True):
    # a.shape (S, P, C, M)
    a = a.copy()
    t = np.nanmean(a, axis=0)
    if argmax: mode = np.argmax(t, axis=axis)
    else: mode = np.argmin(t, axis=axis)
    mode = stats.mode(mode, axis=0, nan_policy="omit").mode
    mode = np.array(list(zip(range(mode.shape[0]), mode)))
    t = a[..., *mode.T]
    for i in range(t.shape[-1]): np.testing.assert_almost_equal(a[..., i, mode[i, 1]], t[..., i])
    return t, mode


def minus_mean_of_rest(y):
    si = []
    for r in range(y.shape[-1]):
        si.append(
            y[..., r]
            - np.mean(np.delete(y, r, axis=-1), axis=-1)
        )
    si = np.array(si)
    si = np.moveaxis(si, 0, -1)
    return si


def make_combined(
    experiment,
    load_fn,
    intensity,
    features,
    response,
    run_id,
    set_reference=False,
    seperator=SEPARATOR,
    **kw,
):
    df = load_fn(
        intensity=intensity,
        features=features,
        run_id=run_id,
        set_reference=set_reference,
        **kw
    )
    df[COMBINATION_CDF] = (
        df[features[1:]].apply(tuple, axis=1)
        .apply(lambda x: seperator.join(x))
        .apply(lambda x: f"{x}{seperator}{experiment}")
    )
    return df[[intensity, features[0], COMBINATION_CDF, *response]] 
