import os
from typing import Optional
from pathlib import Path


class Config():
    """ This will be set to working directory by os.getcwd(). """
    """ Don't change """
    CURRENT_PATH: Optional[Path] = None

    """ File (inside data folder) to use for modeling """
    CSV_PATH: Optional[str] = None

    """ Exogenous """
    INTENSITY: str = "pulse_amplitude"      # Rats
    # INTENSITY: str = "intensity"      # Human

    """ Participant """
    SUBJECT: str = "participant"

    """ Features """
    # FEATURES: list[str] = ["segment", "laterality"]
    FEATURES: list[str] = ["compound_position"]

    """ Endogenous """
    RESPONSE: str = ["auc_3"]     # Rats
    # RESPONSE: str = ["auc_1"]     # Rats
    # RESPONSE: str = ["auc"]     # Human
    # ["LBiceps", "LFCR", "LECR", "LTriceps", "LADM", "LDeltoid", "LBicepsFemoris", "RBiceps"]

    """ Preprocess parameters """
    PREPROCESS_PARAMS: dict[str, int] = {
        "scalar_intensity": 1,
        "scalar_response": [1] * 1,
        "min_observations": 0
    }       # Rats
    # PREPROCESS_PARAMS: dict[str, int] = {
    #     "scalar_intensity": 1,
    #     "scalar_response": [1],
    #     "min_observations": 0
    # }       # Rats
    # PREPROCESS_PARAMS: dict[str, int] = {
    #     "scalar_intensity": 1000,
    #     "scalar_response": [1],
    #     "min_observations": 0
    # }       # Human

    """ MCMC parameters """
    MCMC_PARAMS: dict[str, int] = {
        "num_chains": 4,
        "num_warmup": 4000,
        "num_samples": 6000
    }
