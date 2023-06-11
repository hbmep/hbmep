import os
import logging

import arviz as az
import numpyro

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models import Baseline
from hb_mep.experiments import Experiment, SparseDataExperiment
from hb_mep.models.rats.utils import load_data
from hb_mep.utils import timing

numpyro.set_platform("cpu")
numpyro.set_host_device_count(12)
numpyro.enable_x64()

logger = logging.getLogger(__name__)


@timing
def run_inference_rats(
    config: HBMepConfig = HBMepConfig,
    Model: Baseline = Baseline
) -> None:
    data = DataClass(config)
    data.make_dirs()

    a, b = 1, 4
    subset = range(a, b)

    df, _, _ = load_data(subset, data)
    df, encoder_dict = data.build(df)

    model = Model(config)
    mcmc, posterior_samples = model.run_inference(df=df)

    logger.info(f"Rendering convergence diagnostics ...")
    numpyro_data = az.from_numpyro(mcmc)
    diagnostics = az.summary(data=numpyro_data, hdi_prob=.95)
    save_path = os.path.join(data.reports_path, f"MCMC_{model.name}.csv")
    diagnostics.to_csv(save_path)
    logger.info(f"Saved to {save_path}")

    logger.info(f"Rendering fitted recruitment curves ...")
    fig = model.plot(
        df=df, posterior_samples=posterior_samples, encoder_dict=encoder_dict
    )
    save_path = os.path.join(data.reports_path, f"RC_{model.name}.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved to {save_path}")

    logger.info(f"Rendering posterior predictive checks ...")
    fig = model.predictive_check(
        df=df, posterior_samples=posterior_samples
    )
    save_path = os.path.join(data.reports_path, f"PPC_{model.name}.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved to {save_path}")

    logger.info(f"Finished saving artefacts to {data.reports_path}")
    return


@timing
def run_inference(
    config: HBMepConfig = HBMepConfig,
    model: Baseline = Baseline
) -> None:
    data = DataClass(config)
    data.make_dirs()

    # Load data and preprocess
    df, _ = data.build()

    # Run MCMC inference
    _, posterior_samples = model.run_inference(df=df)

    # Plots
    fig = model.plot(df=df, posterior_samples=posterior_samples)

    # Save artefacts
    save_path = os.path.join(data.reports_path, f"fit_{model.name}.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved artefacts to {save_path}")
    return


@timing
def run_experiment(
    config: HBMepConfig = HBMepConfig,
    experiment: Experiment = SparseDataExperiment
):
    data = DataClass(config)
    data.make_dirs()
    experiment.run()
    return
