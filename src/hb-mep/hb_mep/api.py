import os
import logging

import numpyro

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models import Baseline
from hb_mep.experiments import Experiment, SparseDataExperiment
from hb_mep.utils import timing

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
numpyro.enable_x64()

logger = logging.getLogger(__name__)


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
