from ast import Str
from pathlib import Path


class HBMepConfig():
    # Don't change
    # This will be set to working directory by os.getcwd()
    CURRENT_PATH: Path = None

    # File (present in data folder) to use for modeling
    FNAME: str = 'rats_data_13.csv'

    # Independent feature to study
    INDEPENDENT_FEATURES: list[str] = ['ch_combination']

    # Response MEP
    RESPONSE_MUSCLES: list[str] = ['AUC_Biceps']

    # Preprocess parameters
    PREPROCESS_PARAMS: dict[str, int] = {
        'min_observations': 0,
        'scalar_intensity': 1/10,
        'scalar_mep': 1e7
    }
    ZERO_ONE_THRESHOLDS: list[int] = [0.]

    # MCMC parameters
    MCMC_PARAMS: dict[str, int] = {
        'num_chains': 4,
        'num_warmup': 10000,
        'num_samples': 10000
    }

    # Figure names
    PLOT_FIT: str = 'fit.png'
    PLOT_KDE: str = 'kde.png'

    # Render model filename
    RENDER_FNAME: str = 'rendered_model.png'
