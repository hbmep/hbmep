from hbmep.notebooks.rat.util import get_paths

EXPERIMENT = "L_SHIE"
BUILD_DIR, TOML_PATH, DATA_PATH, MEP_MATRIX_PATH = get_paths(EXPERIMENT)

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
