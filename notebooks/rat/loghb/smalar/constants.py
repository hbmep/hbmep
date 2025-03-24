import os

from hbmep.notebooks.constants import HOME
from hbmep.notebooks.rat.util import get_paths

EXPERIMENT = "C_SMA_LAR"
BUILD_DIR, TOML_PATH, DATA_PATH, MEP_MATRIX_PATH = get_paths(EXPERIMENT)
DATA_PATH_FILTERED = os.path.join(
    HOME,
    "data",
    "hbmep-processed",
    "rat",
    EXPERIMENT,
    "data_filtered.csv"
)

# Laterality
GROUND_BIG = [
    ('-M', '-C5', 'B'),
    ('-LL', '-C5', 'B'),
    ('-L', '-C5', 'B'),
    ('-LM', '-C5', 'B'),

    ('-M', '-C6', 'B'),
    ('-LL', '-C6', 'B'),
    ('-L', '-C6', 'B'),
    ('-LM', '-C6', 'B'),
]
GROUND_SMALL = [
	('-M', '-C5', 'S'),
	('-LL', '-C5', 'S'),
    ('-L', '-C5', 'S'),
	('-LM1', '-C5', 'S'),
	('-LM2', '-C5', 'S'),

	('-M', '-C6', 'S'),
	('-LL', '-C6', 'S'),
	('-L', '-C6', 'S'),
	('-LM1', '-C6', 'S'),
	('-LM2', '-C6', 'S'),
]

NO_GROUND_BIG = [
    ('M-LL', 'C5-C5', 'B'),
    ('M-L', 'C5-C5', 'B'),
    ('M-LM', 'C5-C5', 'B'),

    ('M-LL', 'C6-C6', 'B'),
    ('M-L', 'C6-C6', 'B'),
    ('M-LM', 'C6-C6', 'B'),
]
NO_GROUND_SMALL = [
    ('M-LL', 'C5-C5', 'S'),
    ('M-L', 'C5-C5', 'S'),
    ('M-LM1', 'C5-C5', 'S'),
    ('M-LM2', 'C5-C5', 'S'),

    ('M-LL', 'C6-C6', 'S'),
    ('M-L', 'C6-C6', 'S'),
    ('M-LM1', 'C6-C6', 'S'),
    ('M-LM2', 'C6-C6', 'S'),
]


# Size
GROUND = [
    ('-L', '-C5', 'B'),
    ('-L', '-C5', 'S'),
    ('-L', '-C6', 'B'),
    ('-L', '-C6', 'S'),

    ('-LL', '-C5', 'B'),
    ('-LL', '-C5', 'S'),
    ('-LL', '-C6', 'B'),
    ('-LL', '-C6', 'S'),

    ('-LM', '-C5', 'B'),
    ('-LM1', '-C5', 'S'),
    ('-LM', '-C6', 'B'),
    ('-LM1', '-C6', 'S'),

    ('-M', '-C5', 'B'),
    ('-M', '-C5', 'S'),
    ('-M', '-C6', 'B'),
    ('-M', '-C6', 'S')
]
NO_GROUND = [
    ('M-L', 'C5-C5', 'B'),
    ('M-L', 'C5-C5', 'S'),
    ('M-L', 'C6-C6', 'B'),
    ('M-L', 'C6-C6', 'S'),

    ('M-LL', 'C5-C5', 'B'),
    ('M-LL', 'C5-C5', 'S'),
    ('M-LL', 'C6-C6', 'B'),
    ('M-LL', 'C6-C6', 'S'),

    ('M-LM', 'C5-C5', 'B'),
    ('M-LM1', 'C5-C5', 'S'),
    ('M-LM', 'C6-C6', 'B'),
    ('M-LM1', 'C6-C6', 'S'),
]
