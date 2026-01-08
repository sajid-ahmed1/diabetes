# cleaned dataset
# reminder of the plan
# Impute 0's in Glucose (Mean), BMI (Median), BloodPressure (Median)
# Log Insulin, DiabetesPedigreeFunction, BloodPressure
# Keep Age untouched
# Winsorize Pregnancies, SkinThickness, BloodPressure_log, BMI
# Final DF will be the cleaned one whilst removing the non-logged versions of the logged features
from typing import Any

import numpy as np
import pandas as pd


def clean_data(
    df: pd.DataFrame, stats: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Function to clean the dataframe.
    If stats is None, calculates statistics from df (Training mode).
    If stats is provided, uses them to impute/clip (Test/Production mode).
    """
    df = df.copy()

    # Initialize stats if training
    if stats is None:
        stats = {}
        is_training = True
    else:
        is_training = False

    # 1. Impute (Fix the Zeros)
    # Calculate stats on valid data if training
    if is_training:
        stats["Glucose_mean"] = df.loc[df["Glucose"] != 0, "Glucose"].mean()
        stats["BMI_median"] = df.loc[df["BMI"] != 0, "BMI"].median()
        stats["BP_median"] = df.loc[df["BloodPressure"] != 0, "BloodPressure"].median()

    # Apply imputation using the stored stats
    df["Glucose"] = df["Glucose"].replace(0, stats["Glucose_mean"])
    df["BMI"] = df["BMI"].replace(0, stats["BMI_median"])
    df["BloodPressure"] = df["BloodPressure"].replace(0, stats["BP_median"])

    # 2. Log Transform
    df["Insulin_log"] = np.log1p(df["Insulin"])  # use log1p in case any 0s remain
    df["DPF_log"] = np.log1p(df["DiabetesPedigreeFunction"])
    df["BloodPressure_log"] = np.log1p(df["BloodPressure"])

    # 3. Feature Engineering (Interactions)
    df["Glucose_Insulin_Interaction"] = df["Glucose"] * df["Insulin"]
    df["Age_Insulin_Interaction"] = df["Age"] * df["Insulin"]

    # 4. Winsorize (Clipping)
    # We use clip() with calculated quantiles to ensure consistency between train/test
    cols_to_clip = ["Pregnancies", "SkinThickness", "BloodPressure_log", "BMI"]

    if is_training:
        for col in cols_to_clip:
            # Calculate 99th percentile (equivalent to winsorize limits=[0, 0.01])
            stats[f"{col}_99"] = df[col].quantile(0.99)

    for col in cols_to_clip:
        df[col] = df[col].clip(upper=stats[f"{col}_99"])

    # 5. Final Selection
    final_cols = [
        "Id",
        "Pregnancies",
        "Glucose",
        "SkinThickness",
        "BMI",
        "Age",
        "Insulin_log",
        "DPF_log",
        "BloodPressure_log",
        "Glucose_Insulin_Interaction",
        "Age_Insulin_Interaction",
        "Outcome",
        "sample",
    ]
    # Only keep columns that exist
    existing_cols = [c for c in final_cols if c in df.columns]
    clean_df = df[existing_cols]

    return clean_df, stats
