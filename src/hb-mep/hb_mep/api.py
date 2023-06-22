import os
import logging
from pathlib import Path
from typing import Optional

import arviz as az
import pandas as pd
import numpy as np
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
    data: DataClass,
    model: Baseline,
    reports_path: Path,
    mat: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None
) -> None:
    if mat is not None:
        assert time is not None

    """ Preprocess """
    df, encoder_dict, mat = data.build(df=df, mat=mat)

    """ Run inference """
    mcmc, posterior_samples = model.run_inference(df=df)

    """ Save artefacts """
    logger.info(f"Saving inference data ...")
    numpyro_data = az.from_numpyro(mcmc)
    save_path = os.path.join(reports_path, "mcmc.nc")
    numpyro_data.to_netcdf(save_path)
    logger.info(f"Saved to {save_path}")

    logger.info(f"Rendering convergence diagnostics ...")
    diagnostics = az.summary(data=numpyro_data, hdi_prob=.95)
    save_path = os.path.join(reports_path, f"diagnostics.csv")
    diagnostics.to_csv(save_path)
    logger.info(f"Saved to {save_path}")

    logger.info(f"Evaluating LOO / WAIC scores ...")

    score = az.loo(numpyro_data)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    save_path = os.path.join(reports_path, f"loo.csv")
    score.to_csv(save_path)

    score = az.waic(numpyro_data)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")
    save_path = os.path.join(reports_path, f"waic.csv")
    score.to_csv(save_path)

    logger.info(f"Rendering fitted recruitment curves ...")
    fig = model.plot(
        df=df,
        posterior_samples=posterior_samples,
        encoder_dict=encoder_dict,
        mat=mat,
        time=time
    )
    save_path = os.path.join(reports_path, f"recruitment_curves.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved to {save_path}")

    logger.info(f"Rendering posterior predictive checks ...")
    fig = model.predictive_check(
        df=df, posterior_samples=posterior_samples
    )
    save_path = os.path.join(reports_path, f"posterior_predictive.png")
    fig.savefig(save_path, facecolor="w")
    logger.info(f"Saved to {save_path}")

    logger.info(f"Finished saving artefacts to {reports_path}")
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
