class HBMepConfig():
    # Don't change
    # This will be set to working directory by os.getcwd()
    CURRENT_PATH = None

    # File (present in data folder) to use for modeling
    FNAME = 'rats_data.csv'

    # Segment feature to compare
    SEGMENT_FEATURE = 'level'

    # Preprocess settings
    PREPROCESS_PARAMS = {
        'min_observations': 25,
        'scalar_intensity': 1000,
        'scalar_mep': 1
    }
