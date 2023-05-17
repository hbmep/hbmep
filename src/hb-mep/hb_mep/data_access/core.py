import os
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import jax.numpy as jnp
from jax import random
import pandas as pd
import numpyro.distributions as dist

from hb_mep.config import HBMepConfig
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    DATA_DIR,
    REPORTS_DIR,
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES,
    AUC_MAP
)

logger = logging.getLogger(__name__)


class DataClass:
    def __init__(self, config: HBMepConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

    def make_dirs(self):
        dirs = {
            DATA_DIR: self.data_path,
            REPORTS_DIR: self.reports_path
        }
        for dir in dirs:
            dirs[dir].mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {dir} directory {dirs[dir]}")

    @timing
    def preprocess(
        self,
        df: pd.DataFrame,
        min_observations: int = 25,
        scalar_intensity: float = 1000,
        scalar_response: float = 1
        ) -> tuple[pd.DataFrame, dict, dict[str,  LabelEncoder]]:
        """
        Preprocess data
        """
        # Scale data
        df[INTENSITY] = df[INTENSITY].apply(lambda x: x * scalar_intensity)
        df[RESPONSE] = df[RESPONSE].apply(lambda x: x * scalar_response)

        # Mininum observations constraint
        temp_df = df \
                .groupby(by=[PARTICIPANT] + FEATURES) \
                .size() \
                .to_frame('counts') \
                .reset_index().copy()
        temp_df = temp_df[temp_df.counts >= min_observations].copy()
        keep = list(temp_df[[PARTICIPANT] + FEATURES].apply(tuple, axis=1))
        idx = df[[PARTICIPANT] + FEATURES].apply(tuple, axis=1).isin(keep)
        df = df[idx].copy()

        # Encode participants and features
        encoder_dict = defaultdict(LabelEncoder)
        df[[PARTICIPANT] + FEATURES] = \
            df[[PARTICIPANT] + FEATURES] \
            .apply(lambda x: encoder_dict[x.name].fit_transform(x)).copy()

        df.reset_index(inplace=True, drop=True)
        return df, encoder_dict

    @timing
    def build(self, df: pd.DataFrame = None):
        if df is None:
            fpath = os.path.join(self.data_path, self.config.FNAME)
            logger.info(f"Reading data from {fpath}...")
            df = pd.read_csv(fpath)

        df[f"raw_{RESPONSE}"] = df[RESPONSE]
        logger.info('Processing data ...')
        return self.preprocess(df, **self.config.PREPROCESS_PARAMS)

    # @timing
    # def clean_human_data(
    #     self,
    #     df: pd.DataFrame,
    #     sc_approach: str = "posterior",
    #     muscle: str = "Biceps"
    # ):
    #     """
    #     Clean human data
    #     """
    #     # Validate input
    #     assert sc_approach in ["anterior", "posterior"]

    #     # `sc_electrode` and `sc_electrode_type` are in 1-1 relationship
    #     # # Can be verified using
    #     # df.groupby("sc_electrode")["sc_electrode_type"].apply(lambda x: x.nunique() == 1).all()
    #     # df.groupby("sc_electrode_type")["sc_electrode"].apply(lambda x: x.nunique() == 1).all()
    #     # We can drop either one
    #     df.drop(columns=["sc_electrode"], axis=1, inplace=True)

    #     # Keep `mode` as `research_scs`
    #     df = df[df["mode"]=="research_scs"].copy()

    #     # Keep `sc_depth` as `epidural`
    #     df = df[(df.sc_depth.isin(["epidural"]))].copy()

    #     # # Both `reject_research_scs03` and `reject_research_scs14` must be simultaneously False
    #     # df = df[(df.reject_research_scs03)==False & (df.reject_research_scs14==False)].copy()

    #     # Filter by sc_approach
    #     df = df[df.sc_approach.isin([sc_approach])].copy()

    #     # Remove rows with `sc_laterality` not in ["L", "R", "M"] # RM?
    #     df = df[df.sc_laterality.isin(["L", "R", "M"])].copy()

    #     ## Experiment filters
    #     # Keep `sc_count` equal to 3
    #     df = df[(df.sc_count.isin([3]))].copy()
    #     # Keep `sc_electrode_type` as `handheld`
    #     df = df[(df.sc_electrode_type.isin(["handheld"]))].copy()
    #     # Keep `sc_electrode_configuration` as `RC`
    #     df = df[(df.sc_electrode_configuration.isin(["RC"]))].copy()

    #     # Keep most frequent combination
    #     experiment = [
    #         "sc_laterality",
    #         "sc_count",
    #         "sc_polarity",
    #         "sc_electrode_configuration",
    #         "sc_electrode_type",
    #         "sc_iti"
    #     ]
    #     subset = \
    #         [AUC_MAP["L" + muscle], AUC_MAP["R" + muscle], "sc_current", "sc_level", "sc_cluster_as"]
    #     dropna_subset = subset + experiment

    #     df = df.dropna(subset=dropna_subset, axis="rows", how="any").copy()

    #     df["experiment"] = df[experiment].apply(tuple, axis=1).values

    #     temp_df = df.groupby(["participant"]).experiment.apply(
    #         lambda x: x.value_counts().index[0] if x.value_counts().index[0][0] != "M" \
    #         else x.value_counts().index[1] if len(x.value_counts()) > 1 \
    #         else "M_ONLY"
    #     ).reset_index().copy()

    #     temp_df = temp_df[temp_df.experiment != "M_ONLY"].copy()

    #     assert(temp_df.experiment.apply(lambda x: x[0]).isin(["L", "R"]).all())

    #     participant_to_sc_laterality_map = \
    #         temp_df.apply(lambda x: (x.participant, x.experiment[0]), axis=1).tolist()

    #     participant_to_sc_laterality_map = {
    #         participant: sc_laterality \
    #         for (participant, sc_laterality) in participant_to_sc_laterality_map
    #         }

    #     keep_combinations = temp_df.apply(lambda x: (x.participant, x.experiment), axis=1).tolist()
    #     keep_combinations += \
    #         [(participant, ("M", sc_count, sc_polarity, sc_econfig, sc_etype, sc_iti)) \
    #         for participant, (_, sc_count, sc_polarity, sc_econfig, sc_etype, sc_iti) \
    #         in keep_combinations]

    #     idx = df.apply(lambda x: (x.participant, x.experiment), axis=1).isin(keep_combinations)
    #     df = df[idx].copy()

    #     # Keep most frequent `sc_cluster_as` for each (`participant`, `sc_level`)
    #     keep_combinations = \
    #         df \
    #         .groupby(by=["participant", "sc_level"]) \
    #         .sc_cluster_as.apply(lambda x: x.value_counts().index[0]) \
    #         .reset_index() \
    #         .apply(tuple, axis=1) \
    #         .tolist()

    #     idx = df.apply(lambda x: (x.participant, x.sc_level, x.sc_cluster_as), axis=1).isin(keep_combinations)
    #     df = df[idx].copy()

    #     df[PARTICIPANT] = df["participant"]
    #     df[INTENSITY] = df["sc_current"]
    #     df[FEATURES[0]] = df["sc_level"]
    #     df[FEATURES[1]] = df["sc_laterality"]

    #     df[RESPONSE] = \
    #         df.apply(
    #             lambda x: x[AUC_MAP["L" + muscle]] \
    #             if participant_to_sc_laterality_map[x.participant] == "L"
    #             else x[AUC_MAP["R" + muscle]],
    #             axis=1
    #         )

    #     # Relevant columns
    #     columns = [PARTICIPANT, INTENSITY, RESPONSE] + FEATURES
    #     df = df[columns].copy()

    #     return df

    @timing
    def clean_human_data(
        self,
        df: pd.DataFrame,
        sc_approach: str = "posterior",
        muscle: str = "Triceps"
    ):
        """
        Clean human data
        """
        # Validate input
        assert sc_approach in ["anterior", "posterior"]

        # `sc_electrode` and `sc_electrode_type` are in 1-1 relationship
        # # Can be verified using
        # df.groupby("sc_electrode")["sc_electrode_type"].apply(lambda x: x.nunique() == 1).all()
        # df.groupby("sc_electrode_type")["sc_electrode"].apply(lambda x: x.nunique() == 1).all()
        # We can drop either one
        df.drop(columns=["sc_electrode"], axis=1, inplace=True)

        experiment = [
            "sc_laterality",
            "sc_count",
            "sc_polarity",
            "sc_electrode_configuration",
            "sc_electrode_type",
            "sc_iti"
        ]
        subset = \
            [
                AUC_MAP["L" + muscle], AUC_MAP["R" + muscle], "sc_cluster_as", "sc_current", "sc_level", "sc_cluster_as"
            ]
        dropna_subset = subset + experiment
        df = df.dropna(subset=dropna_subset, axis="rows", how="any").copy()

        # Keep `mode` as `research_scs`
        df = df[df["mode"]=="research_scs"].copy()

        # Keep `sc_depth` as `epidural`
        df = df[(df.sc_depth.isin(["epidural"]))].copy()

        # # Both `reject_research_scs03` and `reject_research_scs14` must be simultaneously False
        # df = df[(df.reject_research_scs03)==False & (df.reject_research_scs14==False)].copy()

        # Filter by sc_approach
        df = df[df.sc_approach.isin([sc_approach])].copy()

        # Remove rows with `sc_laterality` not in ["L", "R", "M"] # RM?
        df = df[df.sc_laterality.isin(["L", "R", "M"])].copy()

        ## Experiment filters
        # Keep `sc_count` equal to 3
        df = df[(df.sc_count.isin([3]))].copy()
        # Keep `sc_electrode_type` as `handheld`
        df = df[(df.sc_electrode_type.isin(["handheld"]))].copy()
        # Keep `sc_electrode_configuration` as `RC`
        df = df[(df.sc_electrode_configuration.isin(["RC"]))].copy()

        keep_combinations = [
            ("cornptio001", "C7", "R"),

            ("cornptio003", "C7", "L"),

            ("cornptio004", "C7", "R"),

            ("cornptio007", "C6", "R"),

            ("cornptio008", "C8", "L"),

            ("cornptio010", "C6", "L"),

            ("cornptio011", "C7", "R"),

            ("cornptio012", "C8", "L"),

            ("cornptio013", "C7", "L"),

            ("cornptio014", "C7", "L"),

            ("cornptio015", "T1", "L"),

            ("cornptio017", "C4", "L"),

            ("scapptio001", "C8", "L")
        ]

        LATERALITY_MAP = {c[0]: c[2] for c in keep_combinations}

        keep_combinations = \
            keep_combinations + [(c[0], c[1], "M") for c in keep_combinations]

        idx = df.apply(lambda x: (x.participant, x.sc_level, x.sc_laterality), axis=1).isin(keep_combinations)
        df = df[idx].copy()

        # cornptio004 Remove `sc_cluster_as`
        remove_combinations = \
            [
                ("cornptio001", "C7", "M", 4),
                ("cornptio004", "C7", "M", 11),
                ("cornptio014", "C7", "M", 11),
                ("cornptio003", "C7", "M", 17),
                ("cornptio003", "C7", "M", 18),
                ("cornptio007", "C6", "R", 10),
                ("cornptio007", "C6", "M", 7)
            ]
        idx = \
            df.apply(lambda x: (x.participant, x.sc_level, x.sc_laterality, x.sc_cluster_as), axis=1) \
            .isin(remove_combinations)
        df = df[~idx].copy()

        df[PARTICIPANT] = df["participant"]
        df[INTENSITY] = df["sc_current"]
        df[FEATURES[0]] = df["sc_level"]
        df[FEATURES[1]] = df["sc_laterality"].apply(lambda x: x if x == "M" else "L")

        df[RESPONSE] = \
            df.apply(
                lambda x: x[AUC_MAP["L" + muscle]] \
                if LATERALITY_MAP[x.participant] == "L"
                else x[AUC_MAP["R" + muscle]],
                axis=1
            )

        # # Relevant columns
        # columns = [PARTICIPANT, INTENSITY, RESPONSE] + FEATURES
        # df = df[columns].copy()

        return df