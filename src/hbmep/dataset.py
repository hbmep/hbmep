import os
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def fit_transform(df, *, features: list[str]) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoder = defaultdict(LabelEncoder)
    df[features] = df[features].apply(lambda x: encoder[x.name].fit_transform(x))
    return df, encoder


def inverse_transform(df, *, encoder: dict[str, LabelEncoder], features: list[str] | None = None) -> pd.DataFrame:
    if features is None: features = list(encoder.keys())
    df[features] = df[features].apply(lambda x: encoder[x.name].inverse_transform(x))
    return df


def make_prediction_dataset(
    df: pd.DataFrame,
    *,
    intensity: str,
    features: list[str],
    num_points: int = 100,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
) -> pd.DataFrame:
    prediction_df = (
        df
        .groupby(by=features)
        .agg({intensity: ["min", "max"]})
        .copy()
    )
    prediction_df.columns = (
        prediction_df
        .columns
        .map(lambda x: x[1])
    )
    prediction_df = prediction_df.reset_index().copy()

    if min_intensity is not None:
        prediction_df["min"] = min_intensity

    if max_intensity is not None:
        prediction_df["max"] = max_intensity

    prediction_df[intensity] = (
        prediction_df[["min", "max"]]
        .apply(tuple, axis=1)
        .apply(lambda x: np.linspace(x[0], x[1], num_points))
    )
    prediction_df = prediction_df.explode(column=intensity)[[intensity] + features].copy()
    prediction_df[intensity] = prediction_df[intensity].astype(float)
    prediction_df = prediction_df.reset_index(drop=True).copy()
    return prediction_df
