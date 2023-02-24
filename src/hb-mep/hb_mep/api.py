import os
import logging

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models.baseline import Baseline
from hb_mep.utils import timing

logger = logging.getLogger(__name__)


@timing
def run_inference(
    model: Baseline,
    config: HBMepConfig = HBMepConfig
) -> None:
    # Load data and preprocess
    data = DataClass(config)
    data.make_dirs()
    df, data_dict, encoders_dict = data.build()

    # Run MCMC inference
    mcmc, posterior_samples = model.sample(data_dict=data_dict)

    # Plots
    fit_fig = model.plot_fit(
        df=df,
        data_dict=data_dict,
        encoders_dict=encoders_dict,
        posterior_samples=posterior_samples
    )
    kde_fig = model.plot_kde(
        data_dict=data_dict, posterior_samples=posterior_samples
    )

    # Save artefacts
    fit_fig.savefig(
        os.path.join(data.reports_path, config.PLOT_FIT),
        facecolor='w'
    )
    kde_fig.savefig(
        os.path.join(data.reports_path, config.PLOT_KDE),
        facecolor='w'
    )
    logger.info(f'Saved artefacts to {data.reports_path}')
