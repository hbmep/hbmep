import os

import pandas as pd
import numpy as np

from constants import SHEET_PATH, DATA_PATH


def _process(data_path, plate):
    plate, skipx, skipy = plate
    x = pd.read_excel(
        data_path, sheet_name=plate, skiprows=skipx[0], nrows=9, header=0
    )
    x = x.iloc[:, skipx[1]:skipx[1] + (12 * 2)].to_numpy()
    y = pd.read_excel(
        data_path, sheet_name=plate, skiprows=skipy[0], nrows=9, header=0
    )
    y = y.iloc[:, skipy[1]:skipy[1] + (12 * 1)].to_numpy()

    conc = x[:, ::2]
    contam = x[:, 1::2]
    indicator = (contam * 0) + contam[0:1, :]
    conc = conc.reshape(-1,)
    contam = contam.reshape(-1,)
    indicator = indicator.reshape(-1,)
    y = y.reshape(-1,)
    
    df = pd.DataFrame(
        np.array([conc, contam, indicator, y]).T,
        columns=["conc", "contam", "id", "y"]
    )
    df["plate"] = plate.replace(" ", "_")
    return df


def process():
    data_path = SHEET_PATH
    plates = [
        ("070825 plate", (2, 2), (42, 2)),
        ("070925 plate", (2, 1), (32, 2))
    ]
    df = [_process(data_path, plate) for plate in plates]
    df = pd.concat(df, ignore_index=True)
    df = df.reset_index(drop=True).copy()

    output_path = DATA_PATH
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return


def main():
    process()


if __name__ == "__main__":
    main()
