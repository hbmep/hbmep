import os

import pandas as pd
import numpy as np
import pandas as pd

HOME = os.getenv("HOME")
DATA_PATH = f"{HOME}/repos/refactor/hbmep/notebooks/im/data/sheet1.xlsx"
BUILD_DIR = f"{HOME}/reports/hbmep/notebooks/im/hbmep-processed"
os.makedirs(BUILD_DIR, exist_ok=True)


data = pd.read_excel(DATA_PATH, sheet_name="Sheet1", skiprows=0)
data.head()
plate_idx = data.iloc[:, 0].astype(str).apply(lambda x: "Plate" in x)
plate_idx = np.where(plate_idx)[0] + 4

df = pd.read_excel(DATA_PATH, sheet_name="Sheet1", skiprows=plate_idx[0], index_col=0)
df.head()

input_columns = [col for col in df.columns if "Derp 1" in str(col)]
input_columns
hue_columns = [col for col in df.columns if "Contaminant" in str(col)]
hue_columns
response_columns = list(df.columns)[0: 12]
response_columns

df[input_columns].head()
df[hue_columns].head()
df[response_columns].head()

dtype = {}
for col in df.columns:
    if col in input_columns or col in response_columns:
        dtype[col] = np.float64
    else: dtype[col] = str


def read_plate(skiprows, plate_id, input, hue, plate, response):
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1", skiprows=skiprows, index_col=0, dtype=dtype, nrows=8) 
    curr_input = df[input_columns].values.reshape(-1,).tolist()  
    input += curr_input
    hue += df[hue_columns].values.reshape(-1,).tolist()
    plate += [f"P{plate_id + 1}"] * len(curr_input)
    response += df[response_columns].values.reshape(-1,).tolist()
    return

input, hue, plate, response = [], [], [], []
plates = [
    read_plate(
        skiprows=idx,
		plate_id=id,
		input=input,
		hue=hue,
		plate=plate,
		response=response
    ) for id, idx in enumerate(plate_idx)
]
df = pd.DataFrame(
    np.array([input, hue, plate, response]).T,
    columns=["conc", "dilution", "plate", "od"],
)
df.conc = df.conc.astype(np.float64)
df.dilution = df.dilution.astype(str)
df.plate = df.plate.astype(str)
df.od = df.od.astype(np.float64)

output_path = os.path.join(BUILD_DIR, "sheet1.csv")
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
