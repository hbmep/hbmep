import logging
from typing import Optional

import pandas as pd
import numpyro
from sklearn.preprocessing import LabelEncoder

from hbmep.config import Config
from hbmep.model import (
    Baseline,
    RectifiedLogistic
)
from hbmep.utils.constants import SIMULATION

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: Config):
        SUPPORTED_MODELS = [Baseline, RectifiedLogistic]

        model_instances = [m(config) for m in SUPPORTED_MODELS]
        model_by_link = {m.link: m for m in model_instances}

        self.model = model_by_link.get(config.LINK)
        logger.info(f"Initialized {self.model.link} model")

    def load(self, df: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        if df is None and self.model.csv_path == SIMULATION:
            df = self.model.simulate()

        df, encoder_dict = self.model.load(df=df)
        return df, encoder_dict

    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        return self.model.run_inference(df)

    def plot(self, df: pd.DataFrame, encoder_dict: dict[str, LabelEncoder]):
        self.model.plot(df=df, encoder_dict=encoder_dict)
        return

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

    def save(self, mcmc: numpyro.infer.mcmc.MCMC):
        self.model.save(mcmc=mcmc)
