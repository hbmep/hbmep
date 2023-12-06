import logging
import tomllib

from hbmep.utils import constants as const

logger = logging.getLogger(__name__)


class Config():
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
            mep_data = cfg.get(const.MEP_DATA, {})

        """ Paths """
        self.TOML_PATH: str = toml_path
        self.CSV_PATH: str = paths[const.CSV_PATH]
        self.BUILD_DIR: str = paths[const.BUILD_DIR]

        """ Regressors and response """
        self.FEATURES: list[str] = vars[const.FEATURES]
        self.INTENSITY: str = vars[const.INTENSITY]
        self.RESPONSE: list[str] = vars[const.RESPONSE]

        """ MCMC parameters """
        self.MCMC_PARAMS: dict[str, int] = {
            "num_chains": mcmc[const.CHAINS],
            "num_warmup": mcmc[const.WARMUP],
            "num_samples": mcmc[const.SAMPLES]
        }

        """ Model """
        self.LINK: str = model[const.LINK]
        self.PRIORS: dict[str, float] = cfg[self.LINK]

        """ MEP data """
        self.MEP_MATRIX_PATH: str | None = mep_data.get(const.MEP_MATRIX_PATH, None)
        self.MEP_RESPONSE: list[str] | None = mep_data.get(const.MEP_RESPONSE, None)
        self.MEP_TIME_RANGE: list[float] | None = mep_data.get(const.MEP_TIME_RANGE, None)
        self.MEP_SIZE_TIME_RANGE: list[float] | None = mep_data.get(const.MEP_SIZE_TIME_RANGE, None)

        """ Aesthetics """
        self.BASE: float = aes[const.BASE]

    def _validate(self):
        # logger.info("Verifying configuration ...")
        # logger.info("Success!")
        pass
