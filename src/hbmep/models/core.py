import logging
from typing import Optional

import pandas as pd
import numpyro

from hbmep.config import MepConfig
from hbmep.models import (
    Baseline,
    RectifiedLogistic
)

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: MepConfig):
        SUPPORTED_MODELS = [Baseline, RectifiedLogistic]

        model_instances = [m(config) for m in SUPPORTED_MODELS]
        model_by_link = {m.link: m for m in model_instances}

        self.model = model_by_link.get(config.LINK)

    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        return self.model.run_inference(df)

    def render_recruitment_curves(
        self,
        df: pd.DataFrame,
        encoder_dict: dict,
        posterior_samples: dict
    ):
        self.model.render_recruitment_curves(
            df=df,
            encoder_dict=encoder_dict,
            posterior_samples=posterior_samples
        )
        return

    def render_predictive_check(
        self,
        df: pd.DataFrame,
        encoder_dict: dict,
        posterior_samples: Optional[dict] = None
    ):
        self.model.render_predictive_check(
            df=df, encoder_dict=encoder_dict,
            posterior_samples=posterior_samples
        )
        return
