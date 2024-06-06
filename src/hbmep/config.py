import logging
import tomllib

from hbmep.utils import constants as const

logger = logging.getLogger(__name__)


class Config():
    def __init__(self, toml_path: str):
        self.TOML_PATH: str = toml_path

        # Load TOML config and validate
        with open(self.TOML_PATH, "rb") as f:
            config = tomllib.load(f)

        self._init(config)
        self._validate(config)

    def _init(self, config):
        paths = config[const.PATHS]
        vars = config[const.VARIABLES]
        mcmc = config[const.MCMC]
        misc = config[const.MISC]
        mep_data = config.get(const.MEP_DATA, {})

        # Paths
        self.CSV_PATH: str = paths[const.CSV_PATH]
        self.BUILD_DIR: str = paths[const.BUILD_DIR]

        # Variables
        self.INTENSITY: str = vars[const.INTENSITY]
        self.FEATURES: list[str] = vars[const.FEATURES]
        self.RESPONSE: list[str] = vars[const.RESPONSE]

        # MCMC parameters
        self.MCMC_PARAMS: dict[str, int] = mcmc

        # MEP data
        self.MEP_DATA: dict = mep_data

        # Misc
        self.BASE: float = misc[const.BASE]

    def _validate(self, config):
        # logger.info("Verifying configuration ...")
        # logger.info("Success!")
        pass
