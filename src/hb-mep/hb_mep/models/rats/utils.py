import os
import glob
import logging
from typing import Optional

import mat73
import numpy as np
import pandas as pd

from hb_mep.data_access import DataClass
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


@timing
def load_data(
    subset: Optional[list[int]],
    data: DataClass = DataClass
):
    df = None

    for i in subset:
        participant = f"amap{i:02}"
        PREFIX = f"rats_data/{participant}/*"

        print(os.path.join(data.data_path, f"{PREFIX}/*auc_table.csv"))
        fpath = glob.glob(os.path.join(data.data_path, f"{PREFIX}/*auc_table.csv"))[0]
        temp_df = pd.read_csv(fpath)

        fpath = glob.glob(os.path.join(data.data_path, f"{PREFIX}/*ep_matrix.mat"))[0]
        data_dict = mat73.loadmat(fpath)
        temp_mat = data_dict["ep_sliced"]

        if df is None:
            time = data_dict["t_sliced"]
        else:
            assert (data_dict["t_sliced"] == time).all()

        temp_df[PARTICIPANT] = participant
        temp_df[FEATURES[0]] = temp_df.channel2_segment
        temp_df[FEATURES[1]] = temp_df.channel2_laterality

        idx = temp_df.channel1_segment.isna()
        temp_df = temp_df[idx].copy()
        temp_df.reset_index(drop=True, inplace=True)

        temp_mat = temp_mat[idx, :, :]

        if df is None:
            df = temp_df.copy()
            mat = temp_mat
        else:
            df = pd.concat([df, temp_df], ignore_index=True).copy()
            mat = np.vstack((mat, temp_mat))

    df.reset_index(drop=True, inplace=True)
    return df, mat, time
