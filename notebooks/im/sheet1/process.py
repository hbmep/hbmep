import os

import pandas as pd
import numpy as np
import pandas as pd

DATA_PATH = "/home/vishu/repos/im/data/sheet1.xlsx"
BUILD_DIR = "/home/vishu/reports/im/sheet1/"
os.makedirs(BUILD_DIR, exist_ok=True)


data = pd.read_excel(DATA_PATH, sheet_name="Sheet1", skiprows=0)
data.head()
plate_idx = data.iloc[:, 0].astype(str).apply(lambda x: "Plate" in x)
plate_idx = np.where(plate_idx)[0] + 4

df = pd.read_excel(DATA_PATH, sheet_name="Sheet1", skiprows=plate_idx[0], index_col=0)
df.head()

input_columns = [col for col in df.columns if "Derp 1" in str(col)]
hue_columns = [col for col in df.columns if "Contaminant" in str(col)]
response_columns = list(df.columns)[0: 12]

df[input_columns].head()
df[response_columns].head()

dtype = {}
for col in df.columns:
    if col in input_columns or col in response_columns:
        dtype[col] = np.float64
    else: dtype[col] = str


def read_plate(skiprows, plate_id, input, response, hue, rowid, plate):
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1", skiprows=skiprows, index_col=0, dtype=dtype, nrows=8) 
    df[input_columns] = df[input_columns].astype(float)
    df[hue_columns] = df[hue_columns].astype(str)
    input += df[input_columns[2:]].values.reshape(-1,).tolist()

    df[input_columns[2:]]

    response += df[response_columns[2:]].values.reshape(-1,).tolist()
    hue += df[hue_columns[2:]].values.reshape(-1,).tolist()
    plate += [plate_id] * df[input_columns[2:]].values.reshape(-1,).shape[0]
    return

input = []
response = []
hue = []
rowid = []
plate = []

plates = [read_plate(idx, id, input, response, hue, rowid, plate) for id, idx in enumerate(plate_idx)]
df = pd.DataFrame(
    np.array([input, response, hue, plate]).T,
    columns=["conc", "od", "dilution", "plate"],
)


df.conc = df.conc.astype(np.float64)
df.od = df.od.astype(np.float64)
df.dilution = df.dilution.astype(str)
df.plate = df.plate.astype(int)

output_path = "/home/vishu/repos/im/data/processed_shee1.csv"
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
