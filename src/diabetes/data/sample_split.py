import hashlib

import numpy as np
import pandas as pd


def create_sample_split(
    df: pd.DataFrame, id_column: str, training_frac: float = 0.8
) -> pd.DataFrame:
    """
    Create sample split based on ID column. Taken from problem set 3

    Parameters
    ----
    df (pd.DataFrame):
        Training data.
    id_column (str):
        Name of ID column.
    training_frac (float):
        Fraction to use for training, by default 0.8.

    Returns
    ----
    pd.DataFrame:
        Training data with sample column containing train/test split
        based on IDs.
    """

    modulo = df[id_column].apply(
        lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
    )

    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")

    return df
