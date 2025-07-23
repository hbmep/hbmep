import os
import pickle
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


def clear_axes(axes):
    [ax.clear() for ax in axes.reshape(-1,)]
    return


def turn_off_ax(ax):
    ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    ax.tick_params(
        axis="both", left=False, bottom=False, labelleft=False,
        labelbottom=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    return


def make_pdf(figs, output_path):
    print("Making pdf...")
    with PdfPages(output_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight') 
            plt.close(fig)
    print(f"Saved to {output_path}")
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
        logger.info(
            f"{mcmc_file} not found."
            + " Attempting to read from model_dict.pkl"
        )
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
            logger.info(
                "Encountered ValueError, trace is below."
                + " Possible issue with unpacking."
            )
            logger.info(e)
        else:
            logger.info("Found model_dict.pkl")

    return df, encoder, posterior, model, mcmc
