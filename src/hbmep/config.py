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
        variables = config[const.VARIABLES]
        mcmc = config[const.MCMC]
        misc = config[const.MISC]
        optional = config.get(const.OPTIONAL, {})

        # Paths
        self.CSV_PATH: str = paths[const.CSV_PATH]
        self.BUILD_DIR: str = paths[const.BUILD_DIR]

        # Variables
        self.INTENSITY: str = variables[const.INTENSITY]
        self.FEATURES: list[str] = variables[const.FEATURES]
        self.RESPONSE: list[str] = variables[const.RESPONSE]

        # MCMC parameters
        self.MCMC_PARAMS: dict[str, int] = mcmc

        # MEP data
        self.MEP_MATRIX_PATH: str | None = optional.get(const.MEP_MATRIX_PATH, None)
        self.MEP_RESPONSE: list[str] | None = optional.get(const.MEP_RESPONSE, None)
        self.MEP_WINDOW: list[float] | None = optional.get(const.MEP_WINDOW, None)
        self.MEP_SIZE_WINDOW: list[float] | None = optional.get(const.MEP_SIZE_WINDOW, None)

        # Misc
        self.BASE: float = misc[const.BASE]

    def _validate(self, config):
        # logger.info("Verifying configuration ...")
        # logger.info("Success!")
        pass
