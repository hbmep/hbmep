import os
import logging

import arviz as az
import pandas as pd
import numpyro

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models import Baseline
from hb_mep.utils import timing

numpyro.set_platform("cpu")
numpyro.set_host_device_count(12)
numpyro.enable_x64()

logger = logging.getLogger(__name__)


@timing
def run_inference(
    df: pd.DataFrame,
    config: HBMepConfig = HBMepConfig,
    data: DataClass = DataClass,
    Model: Baseline = Baseline,
    id: str = "Inference"
) -> None:
    data.make_dirs()
    df, encoder_dict = data.build(df)

    model = Model(config)
    mcmc, posterior_samples = model.run_inference(df=df)

    postfix = f"{model.name}_{id}"

    logger.info(f"Rendering convergence diagnostics ...")
    numpyro_data = az.from_numpyro(mcmc)
    diagnostics = az.summary(data=numpyro_data, hdi_prob=.95)
    save_path = os.path.join(data.reports_path, f"MCMC_{postfix}.csv")
    diagnostics.to_csv(save_path)
    logger.info(f"Saved to {save_path}")

    logger.info(f"Calculating LOO / WAIC scores ...")

    score = az.loo(numpyro_data)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    save_path = os.path.join(data.reports_path, f"LOO_{postfix}.csv")
    score.to_csv(save_path)

    score = az.waic(numpyro_data)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")
    save_path = os.path.join(data.reports_path, f"WAIC_{postfix}.csv")
    score.to_csv(save_path)

    logger.info(f"Rendering fitted recruitment curves ...")
    fig = model.plot(
        df=df, posterior_samples=posterior_samples, encoder_dict=encoder_dict
    )
    save_path = os.path.join(data.reports_path, f"RC_{postfix}.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved to {save_path}")

    logger.info(f"Rendering posterior predictive checks ...")
    fig = model.predictive_check(
        df=df, posterior_samples=posterior_samples
    )
    save_path = os.path.join(data.reports_path, f"PPC_{postfix}.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved to {save_path}")

    logger.info(f"Finished saving artefacts to {data.reports_path}")
    return


# @timing
# def run_experiment(
#     config: HBMepConfig = HBMepConfig,
#     experiment: Experiment = SparseDataExperiment
# ):
#     data = DataClass(config)
#     data.make_dirs()
#     experiment.run()
#     return
