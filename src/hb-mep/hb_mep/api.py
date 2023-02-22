import os
import logging

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models.baseline import Baseline
from hb_mep.utils import (timing, plot_fitted, plot_kde)

logger = logging.getLogger(__name__)


@timing
def run_inference(config: HBMepConfig = HBMepConfig) -> None:
    # Load data and preprocess
    data = DataClass(config)
    data.make_dirs()
    df, data_dict, encoders_dict = data.build()

    # Initialize model
    model = Baseline(config)
    # Run MCMC inference
    mcmc, posterior_samples = model.sample(data_dict=data_dict)

    # Plot fitted curves
    fit_fig = plot_fitted(
        df=df,
        data_dict=data_dict,
        encoders_dict=encoders_dict,
        posterior_samples=posterior_samples
    )
    kde_fig = plot_kde(
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
