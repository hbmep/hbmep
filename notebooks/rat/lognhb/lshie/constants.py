import os

from hbmep.notebooks.constants import INFERENCE_FILE, MODEL_FILE

HOME = os.getenv("HOME")
EXPERIMENT = "L_SHIE"


BUILD_DIR = f"{HOME}/reports/hbmep/notebooks/rat/lognhb/lshie"

TOML_PATH = f"{HOME}/repos/refactor/hbmep/configs/rat/{EXPERIMENT}.toml"
DATA_PATH = f"{HOME}/data/hbmep-processed/rat/{EXPERIMENT}/data.csv"
MEP_MATRIX_PATH = f"{HOME}/data/hbmep-processed/rat/{EXPERIMENT}/mat.npy"
