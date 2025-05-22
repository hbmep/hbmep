import os

from hbmep.notebooks.constants import DATA, REPOS, REPORTS

DATA_PATH = os.path.join(DATA, "im", "sheet1_corrected.csv")
TOML_PATH = os.path.join(REPOS, "hbmep", "configs", "im", "sheet1.toml")
BUILD_DIR = os.path.join(REPORTS, "hbmep", "notebooks", "im", "sheet1")
SIMULATION_DIR = os.path.join(BUILD_DIR, "simulation")
SIMULATION_FACTOR_DIR = os.path.join(SIMULATION_DIR, "dilution_factor")

MAPPING = {
    '1:10': '8__1e-1',
    '1:100': '7__1e-2',
    '1:1,000': '6__1e-3',
    '1:10,000': '5__1e-4',
    '1:100,000': '4__1e-5',
    '1:1,000,000': '3__1e-6',
    '1:10,000,000': '2__1e-7',
    '1:100,000,000': '1__1e-8',
    '0': '0__0',
}

EPS = 1e-3

FACTOR = "dilution_factor"
MIN_CONC = 0.0902185979465152
MAX_CONC = 200.0
REP = "rep"
TOTAL_REPS = 4

FACTORS_SPACE = [1.5, 2, 3, 5, 10]
