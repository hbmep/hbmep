import logging
import tomllib

import pandas as pd

from hbmep.utils import constants as const

logger = logging.getLogger(__name__)


class MepConfig():
    def __init__(self, toml_path: str):
        """ Load TOML config and validate """
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
            self._validate()

            paths = cfg[const.PATHS]
            vars = cfg[const.VARIABLES]
            mcmc = cfg[const.MCMC]
            aes = cfg[const.AESTHETICS]
            model = cfg[const.MODEL]

        """ Paths """
        self.TOML_PATH: str = toml_path
        self.CSV_PATH: str = paths[const.CSV_PATH]
        self.BUILD_DIR: str = paths[const.BUILD_DIR]
        self.RUN_ID: str = paths[const.RUN_ID]

        """ Variables """
        self.SUBJECT: str = vars[const.SUBJECT]
        self.FEATURES: list[str] = vars[const.FEATURES]
        self.INTENSITY: str = vars[const.INTENSITY]
        self.RESPONSE: list[str] = vars[const.RESPONSE]

        """ Preprocess parameters """
        self.PREPROCESS_PARAMS: dict[str, int] = {
            "scalar_intensity": 1,
            "scalar_response": [1] * len(self.RESPONSE),
            "min_observations": 0
        }

        """ MCMC parameters """
        self.MCMC_PARAMS: dict[str, int] = {
            const.NUM_CHAINS: mcmc[const.CHAINS],
            const.NUM_WARMUP: mcmc[const.WARMUP],
            const.NUM_SAMPLES: mcmc[const.SAMPLES]
        }

        """ Aesthetics """
        self.BASE = aes[const.BASE]

        """ Model """
        self.LINK = model[const.LINK]
        self.PRIORS = cfg[self.LINK]

    def _validate(self):
        logger.info("Verifying configuration ...")
        # df = pd.read_csv(self.csv_path)
        # assert set(self.columns) <= set(df.columns)
        logger.info("Success!")
        return
