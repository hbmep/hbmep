import os

HOME = os.getenv("HOME")
DATA = os.path.join(HOME, "data", "hbmep-processed")
REPOS = os.path.join(HOME, "repos", "refactor")
REPORTS = os.path.join(HOME, "reports")

MODEL_FILE = "model.pkl"
INFERENCE_FILE = "inf.pkl"
