import os

from hbmep.notebooks.constants import INFERENCE_FILE, MODEL_FILE

HOME = os.getenv("HOME")
EXPERIMENT = "L_SHIE"

BUILD_DIR = f"{HOME}/reports/hbmep/notebooks/rat/loghb/{EXPERIMENT.lower().replace('_', '')}"

TOML_PATH = f"{HOME}/repos/refactor/hbmep/configs/rat/{EXPERIMENT}.toml"
DATA_PATH = f"{HOME}/data/hbmep-processed/rat/{EXPERIMENT}/data.csv"
MEP_MATRIX_PATH = f"{HOME}/data/hbmep-processed/rat/{EXPERIMENT}/mat.npy"

POSITIONS_MAP = {
    "-C6LC": "-C",
    "C6LC-": "C-",
    "C6LC-C6LX": "C-X",
    "C6LX-C6LC": "X-C",
}
CHARGES_MAP = {
    "50-0-50-100": "Biphasic",
    "20-0-80-25": "Pseudo-Mono"
}

WITH_GROUND = [
    ('-C', 'Biphasic'),
    ('C-', 'Biphasic'),
    ('-C', 'Pseudo-Mono'),
    ('C-', 'Pseudo-Mono'),
]
NO_GROUND = [
    ('C-X', 'Pseudo-Mono'),
    ('C-X', 'Biphasic'),
    ('X-C', 'Pseudo-Mono'),
    ('X-C', 'Biphasic')
]
