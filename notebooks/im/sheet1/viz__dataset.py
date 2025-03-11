import os

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/home/vishu/repos/im/data/processed_shee1.csv"
BUILD_DIR = "/home/vishu/reports/im/sheet1/"
os.makedirs(BUILD_DIR, exist_ok=True)
cmap = sns.color_palette("mako", as_cmap=True)
cmap = sns.diverging_palette(0, 300, as_cmap=True)
cmap = sns.color_palette("icefire", as_cmap=True)


df = pd.read_csv(DATA_PATH)
features = ["plate", "dilution"]
f = df[features].apply(tuple, axis=1)
combinations = f.unique().tolist()

combinations = sorted(combinations)
num_combinations = len(combinations)
nr, nc = num_combinations // 2, 2
fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), constrained_layout=True, squeeze=False, sharex=True, sharey=True)

counter = 0
for c in combinations:
    ax = axes[counter // nc, counter % nc]
    idx = f.isin([c])
    ccdf = df[idx].reset_index(drop=True).copy()
    sns.scatterplot(data=ccdf, x="conc", y="od", ax=ax)
    plate, dil = c
    ax.set_title(f"Plate {plate}, Dilution {dil}")
    counter += 1

for i in range(nr):
    for j in range(nc):
        ax = axes[i, j]
        ax.set_xlabel("")
        ax.set_ylabel("")

ax = axes[0, 0]
ax.set_xscale("log")
ax.set_xlabel("Der p 1 concentration")
ax.set_ylabel("Optical density")

output_path = os.path.join(BUILD_DIR, "dataset.png")
fig.savefig(output_path)
print(f"Saved to {output_path}")
print(0)
