import os

import pandas as pd

from hbmep.notebooks.constants import DATA
from hbmep.notebooks.rat.util import (
    load_circ, load_shie, load_size, load_rcml
)

RUN_ID = "all"
INTENSITY = "pulse_amplitude"

PARTICIPANT = "participant"
RESPONSE = ["LADM", "LBiceps", "LDeltoid", "LECR", "LFCR", "LTriceps"]

SEPARATOR = "___"
COMBINATION_CDF = "combination_cdf"


def process(load_fn, features, experiment):
    df = load_fn(intensity=INTENSITY, features=features, run_id=RUN_ID)
    df[COMBINATION_CDF] = (
        df[features[1:]].apply(tuple, axis=1)
        .apply(lambda x: SEPARATOR.join(x))
        .apply(lambda x: f"{x}{SEPARATOR}{experiment}")
    )
    return df[[INTENSITY, PARTICIPANT, COMBINATION_CDF] + RESPONSE] 


experiments = []

features = ["participant", "compound_position"] 
experiment = "L_CIRC"
experiments.append(experiment)
circ_df = process(load_circ, features, experiment)

features = ["participant", "compound_position", "compound_charge_params"]
experiment = "L_SHIE"
experiments.append(experiment)
shie_df = process(load_shie, features, experiment)

features = ["participant", "segment", "lat", "compound_size"]
experiment = "C_SMA_LAR"
experiments.append(experiment)
size_df = process(load_size, features, experiment)

df = pd.concat([circ_df, shie_df, size_df], ignore_index=True).reset_index(drop=True).copy()
assert df.shape[0] == sum([u.shape[0] for u in [circ_df, shie_df, size_df]])

# output_path = os.path.join(DATA, "rat", f"{SEPARATOR.join(experiments)}.csv")
# df.to_csv(output_path, index=False)
# print(f"Saved to {output_path}")

features = ["participant", "compound_position"]
experiment = "J_RCML"
experiments.append(experiment)
rcml_df = process(load_rcml, features, experiment)

df = pd.concat([circ_df, shie_df, size_df, rcml_df], ignore_index=True).reset_index(drop=True).copy()
assert df.shape[0] == sum([u.shape[0] for u in [circ_df, shie_df, size_df, rcml_df]])

output_path = os.path.join(DATA, "rat", f"{SEPARATOR.join(experiments)}.csv")
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
