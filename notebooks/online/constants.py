import os
USER = os.getenv("USER")

BUILD_DIR = f"/home/{USER}/reports/hbmep/notebooks/online/"
VARIABLES = {
    "intensity": "TMSInt",
    "features": ["participant"],
    "response": ['PKPK_APB']
}
CONFIG = {
    "variables": VARIABLES
}

SIMULATION_DF_PATH = f"/home/{USER}/reports/hbmep/notebooks/online/simulation_df.csv"
SIMULATION_PPD_PATH = f"/home/{USER}/reports/hbmep/notebooks/online/simulation_ppd.pkl"

POSTERIOR_PATH = f"/home/{USER}/reports/hbmep/notebooks/online/posterior.pkl"

TOTAL_SUBJECTS = 32
N_SUBJECTS_SPACE = [1, 2, 4, 8, 16]

TOTAL_PULSES = 64
N_PULSES_SPACE = [16, 24, 32, 40, 48, 56, 64]

TOTAL_REPS = 8
N_REPS_PER_PULSE_SPACE = [1, 4, 8]
