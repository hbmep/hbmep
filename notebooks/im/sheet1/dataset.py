import os

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/home/vishu/repos/im/data/sheet1.xlsx"
BUILD_DIR = "/home/vishu/reports/im/sheet1/"
os.makedirs(BUILD_DIR, exist_ok=True)
cmap = sns.color_palette("mako", as_cmap=True)
cmap = sns.diverging_palette(0, 300, as_cmap=True)
cmap = sns.color_palette("icefire", as_cmap=True)


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
    for kow in range(0, df.shape[0], 2):
        x = df[input_columns[2:]].iloc[row: row + 2].values.reshape(-1,).tolist()
        y = df[response_columns[2:]].iloc[row: row + 2].values.reshape(-1,).tolist()
        label = df[hue_columns[2:]].iloc[row: row + 2].values.reshape(-1,).tolist()
        assert len(np.unique(label)) == 1
        input += x
        response += y
        hue += label
        rowid += [row] * len(x)
        plate += [plate_id] * len(x)


input = []
response = []
hue = []
rowid = []
plate = []

plates = [read_plate(idx, id, input, response, hue, rowid, plate) for id, idx in enumerate(plate_idx)]
df = pd.DataFrame(
    np.array([input, response, hue, rowid, plate]).T,
    columns=["conc", "od", "dilution", "rowid", "plate"],
)
df.conc = df.conc.astype(np.float64)
df.od = df.od.astype(np.float64)
df.dilution = df.dilution.astype(str)
df.rowid = df.rowid.astype(int)
df.plate = df.plate.astype(int)

dilution_map = {}
for cat in df.dilution.unique().tolist():
    match cat:
        case '1:10,000': new_cat = '1:10K'
        case '1:100,000': new_cat = '1:100K'
        case '1:1,000,000': new_cat = '1:1M'
        case '1:10,000,000': new_cat = '1:10M'
        case '1:100,000,000': new_cat = '1:100M'
        case _: new_cat = cat
    dilution_map[cat] = new_cat

df.dilution = df.dilution.replace(dilution_map)
cats = df.dilution.unique().tolist()
dilution_colors = {
    cat: color for cat, color in zip(cats, cmap(np.linspace(0, 1., len(cats))))
}


combinations = df[["dilution", "rowid", "plate"]].apply(tuple, axis=1).unique().tolist()

nr, nc = 1, 1
fig, axes = plt.subplots(nr, nc, figsize=(5, 3), constrained_layout=True, squeeze=False)

ax = axes[0, 0]
ax.clear()
seen = []
for c in combinations:
    dilution, rowid, plate = c
    idx = df[["dilution", "rowid", "plate"]].apply(tuple, axis=1).isin([c])
    ccdf = df[idx].reset_index(drop=True).copy()
    sns.lineplot(x=ccdf.conc, y=ccdf.od, label=dilution if not dilution not in seen else None, color=dilution_colors[dilution])
    seen.append(dilution)

ax.set_xscale("log")
ax.set_xlabel("Der p 1 concentration")
ax.set_ylabel("Optical density")
ax.legend(loc="upper left", title="Contaminant")
# ax.set_xlim(right=10)
output_path = os.path.join(BUILD_DIR, "full.png")
fig.savefig(output_path)
print(f"Saved to {output_path}")
