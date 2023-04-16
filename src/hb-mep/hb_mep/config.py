from pathlib import Path


class HBMepConfig():
    # Don't change
    # This will be set to working directory by os.getcwd()
    CURRENT_PATH: Path = None

    # File (present in data folder) to use for modeling
    FNAME: str = "rats_data_updated.csv"

    # Independent variable
    INTENSITY: str = "intensity"

    # Participant variable
    PARTICIPANT: str = "participant"

    # Dependent variable
    RESPONSE: str = "auc"

    # Study Features
    FEATURES: list[str] = ["segment"]

    # Preprocess parameters
    PREPROCESS_PARAMS: dict[str, int] = {
        "min_observations": 0,
        "scalar_intensity": 1,
        "scalar_response": 1
    }
    ZERO_ONE_THRESHOLDS: list[int] = [0.]

    # MCMC parameters
    MCMC_PARAMS: dict[str, int] = {
        "num_chains": 4,
        "num_warmup": 2000,
        "num_samples": 4000
    }

    # Figure names
    PLOT_FIT: str = "fit.png"
    PLOT_KDE: str = "kde.png"

    # Render model filename
    RENDER_FNAME: str = "rendered_model.png"
