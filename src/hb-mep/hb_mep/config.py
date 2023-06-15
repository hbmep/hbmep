from typing import Optional
from pathlib import Path


class HBMepConfig():
    # This will be set to working directory by os.getcwd(). Don't change
    CURRENT_PATH: Optional[Path] = None

    # File (present in data folder) to use for modeling
    FNAME: Optional[str] = None

    # Independent variable
    INTENSITY: str = "pulse_amplitude"      # Rats
    # INTENSITY: str = "intensity"      # Human

    # Participant variable
    PARTICIPANT: str = "participant"

    # Dependent variable
    RESPONSE: str = ["auc_1", "auc_2"]     # Rats
    # RESPONSE: str = "auc"     # Human

    # Study Features
    FEATURES: list[str] = ["compound_position"]

    # Preprocess parameters
    PREPROCESS_PARAMS: dict[str, int] = {
        "min_observations": 0,
        "scalar_intensity": 1,
        "scalar_response": 1
    }       # Rats
    # PREPROCESS_PARAMS: dict[str, int] = {
    #     "min_observations": 0,
    #     "scalar_intensity": 1000,
    #     "scalar_response": 1
    # }       # Human

    ZERO_ONE_THRESHOLDS: list[int] = [0.]

    # MCMC parameters
    MCMC_PARAMS: dict[str, int] = {
        "num_chains": 4,
        "num_warmup": 4000,
        "num_samples": 6000
    }
