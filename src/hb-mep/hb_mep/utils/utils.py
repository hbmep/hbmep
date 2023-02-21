from pathlib import Path
from typing import Optional
from operator import itemgetter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import numpyro
import h5py
import graphviz
from sklearn.preprocessing import LabelEncoder

from hb_mep.models.baseline import Baseline
from hb_mep.utils.constants import (
    NUM_PARTICIPANTS,
    NUM_SEGMENTS,
    SEGMENTS_PER_PARTICIPANT,
    TOTAL_COMBINATIONS,
    PARTICIPANT_ENCODER,
    SEGMENT_ENCODER,
    INTENSITY,
    MEP_SIZE,
    PARTICIPANT,
    SEGMENT
)

sns.set_theme(style="darkgrid")


def plot(
    df: pd.DataFrame,
    data_dict: dict,
    encoders_dict: dict[str, LabelEncoder],
    posterior_samples: dict,
    keep_muscles: list[str] = ['Biceps'],
    model_function: str = 'relu',
    plot_threshold_kde: bool = True,
    gt: Optional[h5py._hl.files.File] = None,
    ) -> matplotlib.figure.Figure:
    """
    Plot inference results.

    Args:
        df (pd.DataFrame): Data with
        encoders_dict (dict[str, LabelEncoder]): Encoders for participants and levels.
        data_dict (dict): Data dictionary from preprocess.
        posterior_samples (dict): Posterior samples from MCMC.
        keep_muscles (list[str], optional): Plots for only these muscles will be included. Defaults to ['Biceps'].
        model_function (str, optional): Model function to fit on posterior_samples. Must be 'relu', 'sigmoid', or 'softplus'. Defaults to 'relu'.
        plot_threshold_kde (bool, optional): Whether to plot kernel density plot for threshold from posterior_samples. Defaults to True.
        gt (Optional[h5py._hl.files.File], optional): Ground truth for threshold, used when working with simulated data. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure with subplots.
    """
    # Validate arguments
    assert(model_function in ['relu', 'sigmoid', 'softplus'])

    df[keep_muscles] = pd.DataFrame(df[MEP_SIZE].to_list(), columns=keep_muscles)

    n_muscles = len(keep_muscles)
    colors = ['red', 'green', 'blue', 'magenta']
    colors = colors[:n_muscles]

    participant_encoder, level_encoder = itemgetter(PARTICIPANT_ENCODER, SEGMENT_ENCODER)(encoders_dict)
    n_participants, n_levels, levels_per_participant, total_combinations = itemgetter(NUM_PARTICIPANTS, NUM_SEGMENTS, SEGMENTS_PER_PARTICIPANT, TOTAL_COMBINATIONS)(data_dict)

    if plot_threshold_kde:
        fig, axes = plt.subplots(total_combinations*n_muscles, 3, figsize=(12,total_combinations*n_muscles*5))
    else:
        fig, axes = plt.subplots(total_combinations*n_muscles, 2, figsize=(12,total_combinations*n_muscles*5))

    if gt:
        th = np.array(gt['th']).T
        sl = np.array(gt['slope']).T

    posterior_samples['a'] = posterior_samples['a'].reshape(-1, n_levels, n_participants, n_muscles)
    posterior_samples['b'] = posterior_samples['b'].reshape(-1, n_levels, n_participants, n_muscles)

    mean_a = posterior_samples['a'].mean(axis=0)
    mean_b = posterior_samples['b'].mean(axis=0)

    k = 0
    for i in range(n_participants):
        for j in levels_per_participant[i]:
            for m, col in enumerate(keep_muscles):
                if participant_encoder is not None and level_encoder is not None:
                    axes[k+m, 0].set_title(f'Actual: Participant:{i}, Level:{j}, {col}')
                    axes[k+m, 1].set_title(f'Fitted: Participant:{i}, Level:{j}, {col}')
                else:
                    axes[k+m, 0].set_title(f'Actual: Participant:{i+1}, Level:{j+1}')
                    axes[k+m, 1].set_title(f'Fitted: Participant:{i}, Level:{j}')

                sns.scatterplot(data=df[(df.participant==i) & (df.level==j)], x='intensity', y=col, ax=axes[k+m, 0])
                sns.scatterplot(data=df[(df.participant==i) & (df.level==j)], x='intensity', y=col, ax=axes[k+m, 1], alpha=.4)

                x_val = np.linspace(0, 15, 100)

                if model_function == 'relu':
                    y_val = jax.nn.relu(mean_b[j,i,m] * (x_val - mean_a[j,i,m]))
                elif model_function == 'sigmoid':
                    y_val = jax.nn.sigmoid(mean_b[j,i,m] * (x_val - mean_a[j,i,m]))
                else:
                    y_val = jax.nn.softplus(mean_b[j,i,m] * (x_val - mean_a[j,i,m]))

                sns.lineplot(x=x_val, y=y_val, ax=axes[k+m, 1], color=colors[m], alpha=0.4, label=f'Mean Posterior {col}')

                if plot_threshold_kde:
                    sns.scatterplot(data=df[(df.participant==i) & (df.level==j)], x='intensity', y=col, ax=axes[k+m, 2])
                    sns.kdeplot(x=posterior_samples['a'][:,j,i,m], ax=axes[k+m, 1], color='green')
                    sns.lineplot(x=x_val, y=y_val, ax=axes[k+m, 2], color=colors[m], alpha=0.4, label=f'Mean Posterior {col}')

                if gt:
                    y_val_gt = jax.nn.relu(sl[i,j] * (x_val - th[i,j]))
                    sns.lineplot(x=x_val, y=y_val, ax=axes[k+m, 2], color=colors[m], alpha=0.4, label='GT')

            k += n_muscles

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    return fig

def render_model(
    model: Baseline,
    data_dict: dict,
    filename: Optional[Path] = None
    ) -> graphviz.graphs.Digraph:
    """
    Render NumPyro model and save resultant graph.

    Args:
        model (model): NumPyro model for rendering.
        data_dict (dict): Data dictionary containing model parameters for rendering.
        filename (Optional[Path], optional): Target destination for saving rendered graph. Defaults to None.

    Returns:
        graphviz.graphs.Digraph: Rendered graph.
    """
    # Retrieve numpyro model parameters from data dictionary for rendering
    intensity, participant, level, mep_size = itemgetter('intensity', 'participant', 'level', 'mep_size')(data_dict)
    return numpyro.render_model(model, model_args=(intensity, participant, level, mep_size), filename=filename)