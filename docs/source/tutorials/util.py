import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from numpyro.diagnostics import hpdi

from hbmep.util import generate_response_colors
import hbmep as mep


def plot(
    df,
    *,
    intensity,
    features,
    response,
    encoder=None,
    posterior=None,
    prediction_df=None,
    predictive=None,
    predictive_var=mep.site.mu,
    predictive_hdi_var=None,
    predictive_hdi_prob=0.0,
    threshold_var=mep.site.a,
    threshold_hdi_prob=0.0,
):

    df_features = df[features].apply(tuple, axis=1)
    combinations = sorted(df_features.unique().tolist())
    num_combinations = len(combinations)
    num_response = len(response)
    colors = generate_response_colors(num_response)

    nr, nc = num_combinations, num_response
    kr, kc = 1, 1
    heights = [1] * nr
    figsize = (6, 6)

    show_curves = prediction_df is not None and predictive is not None
    show_threshold = posterior is not None and threshold_var in posterior

    if show_curves:
        kr = 2
        heights = [1, .5] * nr
        figsize = (6, 8)
        pred_features = prediction_df[features].apply(tuple, axis=1)
        curve_samples = predictive[predictive_var]
        hdi_samples = (
            predictive[predictive_hdi_var]
            if predictive_hdi_var is not None
            else None
        )

    if show_threshold:
        threshold_samples = posterior[threshold_var]

    fig, axes = plt.subplots(
        *(nr * kr, nc * kc), figsize=figsize, constrained_layout=True,
        squeeze=False, sharex=True, height_ratios=heights
    )

    for i in range(num_combinations):
        combination = combinations[i]
        idx = df_features.isin([combination])
        curr_df = df[idx].reset_index(drop=True).copy()

        if show_curves:
            pred_idx = pred_features.isin([combination])
            curr_pred = prediction_df[pred_idx].reset_index(drop=True).copy()
            curr_curve = curve_samples[:, pred_idx]
            curr_hdi = hdi_samples[:, pred_idx] if hdi_samples is not None else None

        for j in range(num_response):
            ax = axes[i * kr, j * kc]
            x = curr_df[intensity]
            response_name = response[j]
            y = curr_df[response_name]
            sns.scatterplot(x=x, y=y, ax=ax, color=colors[j], s=25)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if not i:
                ax.set_title(response_name, fontsize=12)
            if not j:
                if encoder is None:
                    ax.set_ylabel(", ".join(combination), fontsize=12)
                else:
                    combination_inv = tuple(
                        encoder[features[0]].inverse_transform([u])[0]
                        for u in combination
                    )
                    ax.set_ylabel(f", ".join(combination_inv), fontsize=12)

            if show_curves:
                x = curr_pred[intensity]
                y = curr_curve[..., j].mean(axis=0)

                if predictive_hdi_prob > 0 and curr_hdi is not None:
                    band = hpdi(curr_hdi[..., j], prob=predictive_hdi_prob)
                    ax.fill_between(
                        x,
                        band[0],
                        band[1],
                        color=colors[j],
                        alpha=.15
                    )

                sns.lineplot(x=x, y=y, ax=ax, color=colors[j])

                ax = axes[i * kr + 1, j * kc]
                if show_threshold:
                    samples = threshold_samples[:, *combination, j]
                    sns.kdeplot(samples, color=colors[j], ax=ax)

                    if threshold_hdi_prob > 0:
                        band = hpdi(samples, prob=threshold_hdi_prob)
                        ax.axvline(
                            samples.mean(), linestyle="--", color=colors[j],
                            label="Point estimate"
                        )
                        ax.axvline(
                            band[0], linestyle="--", color="black", alpha=.4,
                            label=f"{int(100 * threshold_hdi_prob)}% HPDI"
                        )
                        ax.axvline(
                            band[1], linestyle="--", color="black", alpha=.4
                        )

                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.text(
                    0.02, 0.7, (*combination, j), fontsize=10,
                    va="bottom", ha="left", transform=ax.transAxes
                )

        miny, maxy = float('inf'), float('-inf')
        for j in range(num_response):
            ax = axes[i * kr, j * kc]
            lo, hi = ax.get_ylim()
            miny = min(miny, lo)
            maxy = max(maxy, hi)
            ax.sharey(axes[i * kr, 0])
        ax = axes[i * kr, 0]
        ax.set_ylim(miny, maxy)

    for i in range(nr):
        for j in range(nc):
            ax = axes[i * kr, j * kc]
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis="x", labelbottom=True)

            if show_curves:
                ax = axes[i * kr + 1, j * kc]
                ax.spines[['right', 'top']].set_visible(False)
                ax.tick_params(axis="both", left=False, labelbottom=True, labelleft=False)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()
