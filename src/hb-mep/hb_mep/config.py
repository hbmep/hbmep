class HBMepConfig():
    # Don't change
    # This will be set to working directory by os.getcwd()
    CURRENT_PATH = None

    # File (present in data folder) to use for modeling
    FNAME = 'rats_data.csv'

    # Segment feature to compare
    SEGMENT_FEATURE = 'level'

    # Preprocess parameters
    PREPROCESS_PARAMS = {
        'min_observations': 20,
        'scalar_intensity': 1/30,
        'scalar_mep': 1e7
    }

    # MCMC parameters
    MCMC_PARAMS = {
        'num_chains': 4,
        'num_warmup': 10000,
        'num_samples': 10000
    }

    # File name to save fit plot
    PLOT_FIT = 'fitted.png'
    # File name to save kde plot
    PLOT_KDE = 'kde.png'

    # Render model filename
    RENDER_FNAME = 'rendered_model.png'
