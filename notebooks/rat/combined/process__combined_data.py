import os

import pandas as pd

from hbmep.notebooks.constants import DATA
from hbmep.notebooks.rat.util import (
    load_circ, load_shie, load_size, load_rcml, make_combined
)

RUN_ID = "all"
INTENSITY = "pulse_amplitude"

PARTICIPANT = "participant"
RESPONSE = ["LADM", "LBiceps", "LDeltoid", "LECR", "LFCR", "LTriceps"]


experiments = []

features = ["participant", "compound_position"] 
experiment = "L_CIRC"
experiments.append(experiment)
circ_df = make_combined(
    experiment, load_circ, INTENSITY, features, RESPONSE, RUN_ID
)

features = ["participant", "compound_position", "compound_charge_params"]
experiment = "L_SHIE"
experiments.append(experiment)
shie_df = make_combined(
    experiment, load_shie, INTENSITY, features, RESPONSE, RUN_ID
)

features = ["participant", "segment", "lat", "compound_size"]
experiment = "C_SMA_LAR"
experiments.append(experiment)
size_df = make_combined(
    experiment, load_size, INTENSITY, features, RESPONSE, RUN_ID
)


df = pd.concat([circ_df, shie_df, size_df], ignore_index=True).reset_index(drop=True).copy()
assert df.shape[0] == sum([u.shape[0] for u in [circ_df, shie_df, size_df]])

# output_path = os.path.join(DATA, "rat", f"{SEPARATOR.join(experiments)}.csv")
# df.to_csv(output_path, index=False)
# print(f"Saved to {output_path}")

# output_path = os.path.join(DATA, "rat", f"{'___'.join(experiments)}___combined.csv")
# df.to_csv(output_path, index=False)
# print(f"Saved to {output_path}")

features = ["participant", "compound_position"]
experiment = "J_RCML"
experiments.append(experiment)
rcml_df = make_combined(
    experiment, load_rcml, INTENSITY, features, RESPONSE, RUN_ID
)

df = pd.concat([circ_df, shie_df, size_df, rcml_df], ignore_index=True).reset_index(drop=True).copy()
assert df.shape[0] == sum([u.shape[0] for u in [circ_df, shie_df, size_df, rcml_df]])

# compare_df = pd.read_csv("/home/vishu/data/hbmep-processed/rat/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML.csv")
# pd.testing.assert_frame_equal(df, compare_df)

# output_path = os.path.join(DATA, "rat", f"{SEPARATOR.join(experiments)}.csv")
# df.to_csv(output_path, index=False)
# print(f"Saved to {output_path}")
