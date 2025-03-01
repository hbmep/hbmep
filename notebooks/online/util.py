import logging
import numpy as np
from constants import TOTAL_PULSES, N_PULSES_SPACE

logger = logging.getLogger(__name__)


def generate_nested_pulses(simulation_df):
    all_pulses = simulation_df["TMSInt"].unique()
    all_pulses = np.sort(all_pulses)

    assert TOTAL_PULSES == all_pulses.shape[0]
    assert TOTAL_PULSES == N_PULSES_SPACE[-1]

    pulses_map = {}
    pulses_map[TOTAL_PULSES] = \
        np.arange(0, TOTAL_PULSES, 1).astype(int).tolist()

    for i in range(len(N_PULSES_SPACE) - 2, -1, -1):
        n_pulses = N_PULSES_SPACE[i]
        subsample_from = pulses_map[N_PULSES_SPACE[i + 1]]
        ind = \
            np.round(np.linspace(0, len(subsample_from) - 1, n_pulses)) \
            .astype(int).tolist()
        pulses_map[n_pulses] = np.array(subsample_from)[ind].tolist()

    for i in range(len(N_PULSES_SPACE) - 1, -1, -1):
        n_pulses = N_PULSES_SPACE[i]
        pulses_map[n_pulses] = list(
            all_pulses[pulses_map[n_pulses]]
        )
        assert set(pulses_map[n_pulses]) <= set(all_pulses)
        assert len(pulses_map[n_pulses]) == n_pulses
        if n_pulses != TOTAL_PULSES:
            assert set(pulses_map[n_pulses]) <= set(
                pulses_map[N_PULSES_SPACE[i + 1]]
            )

    return pulses_map
