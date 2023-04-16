import logging
from time import time
from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from hb_mep.utils.constants import (
    INTENSITY,
    RESPONSE_MUSCLES,
    PARTICIPANT,
    INDEPENDENT_FEATURES,
    MUSCLE_TO_AUC_MAP
)

logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_taken = te - ts
        hours_taken = time_taken // (60 * 60)
        time_taken %= (60 * 60)
        minutes_taken = time_taken // 60
        time_taken %= 60
        seconds_taken = time_taken % 60
        if hours_taken:
            message = \
                f"func:{f.__name__} took: {hours_taken:0.0f} hr and " + \
                f"{minutes_taken:0.0f} min"
        elif minutes_taken:
            message = \
                f"func:{f.__name__} took: {minutes_taken:0.0f} min and " + \
                f"{seconds_taken:0.2f} sec"
        else:
            message = f"func:{f.__name__} took: {seconds_taken:0.2f} sec"
        logger.info(message)
        return result
    return wrap


@timing
def plot(df: pd.DataFrame):
    combinations = \
        df \
        .groupby(by=[PARTICIPANT] + INDEPENDENT_FEATURES) \
        .size() \
        .to_frame('counts') \
        .reset_index().copy()
    combinations = combinations[[PARTICIPANT] + INDEPENDENT_FEATURES].apply(tuple, axis=1).tolist()
    n_combinations = len(combinations)

    fig, axes = plt.subplots(n_combinations, 1, figsize=(8, n_combinations * 3))

    for i, c in enumerate(combinations):
        temp_df = \
            df[df[[PARTICIPANT] + INDEPENDENT_FEATURES] \
            .apply(tuple, axis=1) \
            .isin([c])] \
            .reset_index(drop=True) \
            .copy()

        sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE_MUSCLES[0], ax=axes[i])
        axes[i].set_title(f'Actual: Combination:{c}, {RESPONSE_MUSCLES[0]}')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    return fig


def plot_emg_data(
    df: pd.DataFrame,
    mat: np.ndarray,
    time: np.ndarray,
    metadata: dict,
    muscle: str = "auc_1",
    scalar: int = 80
):
    combinations = \
        df \
        .groupby(by=["participant", "compound_position"]) \
        .size() \
        .to_frame('counts') \
        .reset_index().copy()
    combinations = combinations[["participant", "compound_position"]].apply(tuple, axis=1).tolist()
    total_combinations = len(combinations)

    fig, axes = plt.subplots(total_combinations, 2, figsize=(12, total_combinations * 3), constrained_layout=True)

    for i, c in enumerate(combinations):
        idx = (df.participant == c[0]) & (df.compound_position == c[1])
        temp_df = df[idx]
        temp_mat = mat[idx]

        intensity = temp_df.pulse_amplitude.to_numpy()
        intensity = intensity
        auc = temp_df[muscle].to_numpy()
        signal_mat = (temp_mat / scalar) + intensity.reshape(-1, 1, 1)

        time_minmax = metadata[c[0]]["auc"]["t_slice_minmax"]

        sns.scatterplot(x=intensity, y=auc, ax=axes[i, 1])
        axes[i, 0].plot(signal_mat[:, :, 0].T, time, color="green", alpha=.4)

        axes[i, 0].axhline(y=time_minmax[0], color="red", alpha=.6, linestyle="dashed", label="AUC Window")
        axes[i, 0].axhline(y=time_minmax[1], color="red", alpha=.6, linestyle="dashed")

        axes[i, 1].set_xlabel("Pulse Amplitude $(µA)$")
        axes[i, 1].set_ylabel(r"AUC $(µV - sec)$")

        axes[i, 0].set_xlabel("Pulse Amplitude $(µA)$")
        axes[i, 0].set_ylabel("Time $(sec)$")

        axes[i, 0].set_title(f"Animal: {c[0]}, Position: {c[1]}")
        # axes[i, 0].set_title(f"Motor Evoked Potential")
        axes[i, 0].set_ylim(bottom=-0.001, top=0.02)

        axes[i, 0].legend(loc="upper right")
        axes[i, 1].legend(loc="upper right")

    return fig


# def simulated_data(n_participant: int = 20, n_independent: int = 2):
#     seed = random.PRNGKey(0)

#     a_mean_global_scale = 5.18
#     a_scale_global_scale = 1.47

#     b_mean_global_scale = 4.78
#     b_scale_global_scale = 1.60

#     lo_scale_global_scale = 0.1
#     g_scale_global_scale = 0.001

#     noise_offset_scale_global_scale = 0.02
#     noise_slope_scale_global_scale = 0.1

#     n_participant = 20
#     n_independent = 2

#     a_mean = dist.HalfNormal(a_mean_global_scale).sample(seed, sample_shape=(n_independent,))
#     a_scale = dist.HalfNormal(a_scale_global_scale).sample(seed, sample_shape=(n_independent,))

#     b_mean = dist.HalfNormal(b_mean_global_scale).sample(seed, sample_shape=(n_independent,))
#     b_scale = dist.HalfNormal(b_scale_global_scale).sample(seed, sample_shape=(n_independent,))

#     lo_scale = dist.HalfNormal(lo_scale_global_scale).sample(seed, sample_shape=(n_independent,))
#     g_scale = dist.HalfCauchy(g_scale_global_scale).sample(seed, sample_shape=(n_independent,))

#     noise_offset_scale = dist.HalfCauchy(noise_offset_scale_global_scale).sample(seed, sample_shape=(n_independent,))
#     noise_slope_scale = dist.HalfCauchy(noise_slope_scale_global_scale).sample(seed, sample_shape=(n_independent,))

#     Rho = dist.LKJ(n_independent, concentration=.5).sample(seed, (1,))[0]

#     a_cov = jnp.outer(a_scale, a_scale) * Rho

#     a = dist.MultivariateNormal(a_mean, a_cov).sample(seed, (n_participant,))
#     b = dist.Normal(b_mean, b_scale).sample(seed, (n_participant,))

#     lo = dist.HalfNormal(lo_scale).sample(seed, (n_participant,))
#     g = dist.HalfCauchy(g_scale).sample(seed, (n_participant, ))

#     noise_offset = dist.HalfCauchy(noise_offset_scale).sample(seed, (n_participant,))
#     noise_slope = dist.HalfCauchy(noise_slope_scale).sample(seed, (n_participant * 1000,))

#     noise_slope = noise_slope[((noise_slope > 0.3).all(axis=1)) & ((noise_slope < 2).all(axis=1)), :]
#     idx = random.randint(seed, (n_participant,), 0, noise_slope.shape[0])
#     noise_slope = noise_slope[idx]

#     n_points = 40
#     n_samples = 2
#     x = jnp.linspace(0, 10, n_points)
#     df = None

#     for i in range(n_participant):
#         for j in range(n_independent):
#             participant = jnp.repeat(i, n_points)
#             independent = jnp.repeat(j, n_points)

#             mean = \
#                 -jnp.log(jnp.maximum(g[i, j], jnp.exp(-jnp.maximum(lo[i, j], b[i, j] * (x - a[i, j])))))

#             noise = noise_offset[i, j] + noise_slope[i, j] * mean

#             y = dist.TruncatedNormal(mean, noise, low=0).sample(seed, (n_samples,)).mean(axis=0)

#             if df is None:
#                 df = pd.DataFrame(
#                     jnp.array([participant, independent, x, y.reshape(-1,)]).T,
#                     columns=['participant', 'ch_combination', 'intensity', 'AUC_Biceps']
#                 )
#             else:
#                 temp = pd.DataFrame(
#                     jnp.array([participant, independent, x, y.reshape(-1,)]).T,
#                     columns=['participant', 'ch_combination', 'intensity', 'AUC_Biceps']
#                 )
#                 df = pd.concat([df, temp], ignore_index=True).copy()
#     return df


def clean_human_data(
    df: pd.DataFrame,
    sc_approach: str = 'posterior',
    keep_muscles: list[str] = ['Biceps'],
    keep_combinations_with_both_lateral_and_midline_data: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean real data collected from patients.
    Args:
        df (pd.DataFrame): Data to be cleaned.
        sc_approach (str, optional): Setting for sc_approach. Must be either 'posterior' or 'anterior'. Defaults to 'posterior'.
        keep_muscles (list[str], optional): Data for only these muscles will be kept. Defaults to ['Biceps'].
        keep_combinations_with_both_lateral_and_midline_data (bool, optional): Whether to keep combinations with both lateral and midline data. Defaults to False.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned midline and lateral data.
    """
    # Validate arguments
    assert(sc_approach in ['posterior', 'anterior'])

    # sc_electrode and sc_electrode_type are in 1-1 relation, we can drop sc_electrode
    df.drop(columns=['sc_electrode'], axis=1, inplace=True)

    # Keep only rows with mode = research_scs
    df = df[df['mode']=='research_scs'].copy()

    # Get rid of artefacts
    ## re-check
    df = df[(df.reject_research_scs03)==False & (df.reject_research_scs14==False)].copy()

    # Select sc_approach
    if sc_approach == 'posterior':
        df = df[df.sc_approach.isin(['posterior'])].copy()

    else:
        df = df[df.sc_approach.isin(['anterior'])].copy()

    # Get rid of rows with sc_laterality not in ['L', 'R', 'M'] # RM?
    df = df[df.sc_laterality.isin(['L', 'R', 'M'])].copy()

    # Epidural
    df = df[(df.sc_depth.isin(['epidural']))].copy()

    # Subset most frequent combination for the following columns
    majority_columns = ['sc_laterality', 'sc_count', 'sc_polarity', 'sc_electrode_configuration', 'sc_electrode_type', 'sc_iti']
    subset = ['sc_current', 'sc_cluster_as'] + majority_columns
    df = df.dropna(subset=subset, axis='rows', how='any').copy()
    df['temp'] = df[majority_columns].apply(tuple, axis=1).values

    t = df.groupby(['participant']).temp.apply \
        (lambda x: x.value_counts().index[0] if x.value_counts().index[0][0] != 'M' \
            else x.value_counts().index[1] if len(x.value_counts()) > 1 \
                else 'only_M').reset_index().copy()

    assert(t.participant.is_unique)
    t['temp2'] = t.temp.apply(lambda x: x[0])
    t = t[t.temp2.isin(['L', 'R'])].copy()

    participant_to_sc_laterality_map = list(zip(t.participant, t.temp2))
    participant_to_sc_laterality_map = {
        participant:sc_laterality for (participant, sc_laterality) in participant_to_sc_laterality_map
        }

    keep = list(zip(t.participant, t.temp))
    keep += [(participant, ('M', sc_count, sc_polarity, sc_config, sc_etype, sc_iti)) \
        for participant, (sc_laterality, sc_count, sc_polarity, sc_config, sc_etype, sc_iti) in keep]

    df['temp'] = list(zip(df.participant, df.temp))
    df = df[df.temp.isin(keep)].copy()
    df['temp'] = list(zip(df.participant, df.auc03, df.auc14))

    # Mep size
    for col in keep_muscles:
        left_muscle_auc, right_muscle_auc = MUSCLE_TO_AUC_MAP['L' + col], MUSCLE_TO_AUC_MAP['R' + col]
        df['temp'] = list(zip(df.participant, df[left_muscle_auc], df[right_muscle_auc]))
        df[col] = df.temp.apply(lambda x: x[1] if participant_to_sc_laterality_map[x[0]] == 'L' else x[2])

    # Subset most frequent sc_cluster_as type for each (participant, level)
    t = df.groupby(by=['participant', 'sc_level']).sc_cluster_as.apply(lambda x: x.value_counts().index[0]).reset_index().copy()
    keep = list(zip(t.participant, t.sc_level, t.sc_cluster_as))
    df['temp'] = list(zip(df.participant, df.sc_level, df.sc_cluster_as))
    df = df[df.temp.isin(keep)].copy()

    # Drop NAs
    subset =  ['participant', 'sc_level', 'sc_laterality', 'sc_current'] + keep_muscles
    df.dropna(subset=subset, how='any', inplace=True)
    df['mep_size'] = df[keep_muscles].apply(tuple, axis=1).values

    # Keep only relevant columns for further processing and modeling 
    keep = ['participant', 'sc_level', 'sc_laterality', 'sc_current', 'mep_size'] 
    df = df.drop(columns=[col for col in df.columns if col not in keep]).copy()

    # Rename columns
    columns_mapping = {'participant':'participant', 'sc_level':'level', 'sc_laterality':'laterality', 'sc_current':'intensity'}
    df = df.rename(columns = columns_mapping).copy()

    # Rearrange columns
    subset = ['mep_size', 'participant', 'level', 'intensity', 'laterality']
    df = df[subset].copy()

    # Lateral subset
    lateral = df[df.laterality.isin(['L', 'R'])].copy()
    lateral.drop(columns=['laterality'], axis='columns', inplace=True)
    lateral.reset_index(inplace=True, drop=True)

    # Midline subset
    midline = df[df.laterality.isin(['M'])].copy()
    midline.drop(columns=['laterality'], axis='columns', inplace=True)
    midline.reset_index(inplace=True, drop=True)

    if keep_combinations_with_both_lateral_and_midline_data:
        lateral['temp'] = list(zip(lateral.participant, lateral.level))
        lat = set(lateral.temp)

        midline['temp'] = list(zip(midline.participant, midline.level))
        mid = set(midline.temp)

        keep = list(lat.intersection(mid))

        lateral = lateral[lateral.temp.isin(keep)].drop(['temp'], axis=1).copy()
        lateral.reset_index(inplace=True, drop=True)

        midline = midline[midline.temp.isin(keep)].drop(['temp'], axis=1).copy()
        midline.reset_index(inplace=True, drop=True)

    midline['method'] = 'midline'
    lateral['method'] = 'lateral'

    rdf = pd.concat([lateral, midline], ignore_index=True).copy()
    rdf.mep_size = rdf.mep_size.apply(lambda x: x[0])

    return rdf
