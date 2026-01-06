# cleaned dataset
# reminder of the plan
# Impute 0's in Glucose (Mean), BMI (Median), BloodPressure (Median)
# Log Insulin, DiabetesPedigreeFunction, BloodPressure
# Keep Age untouched
# Winsorize Pregnancies, SkinThickness, BloodPressure_log, BMI
# Final DF will be the cleaned one whilst removing the non-logged versions of the logged features
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean the dataframe
    """
    # 1. Impute (Fix the Zeros)
    df["Glucose"] = df["Glucose"].replace(0, df["Glucose"][df["Glucose"] != 0].mean())
    df["BMI"] = df["BMI"].replace(0, df["BMI"][df["BMI"] != 0].median())
    df["BloodPressure"] = df["BloodPressure"].replace(
        0, df["BloodPressure"][df["BloodPressure"] != 0].median()
    )

    # 2. Log Transform
    df["Insulin_log"] = np.log1p(df["Insulin"])  # use log1p in case any 0s remain
    df["DPF_log"] = np.log1p(df["DiabetesPedigreeFunction"])
    df["BloodPressure_log"] = np.log1p(df["BloodPressure"])

    # 3. Winsorize (using scipy)
    from scipy.stats.mstats import winsorize

    df["Pregnancies"] = winsorize(df["Pregnancies"], limits=[0, 0.01])
    df["SkinThickness"] = winsorize(df["SkinThickness"], limits=[0, 0.01])
    df["BloodPressure_log"] = winsorize(df["BloodPressure_log"], limits=[0, 0.01])
    df["BMI"] = winsorize(df["BMI"], limits=[0, 0.01])

    # 4. Final Selection
    final_cols = [
        "Pregnancies",
        "Glucose",
        "SkinThickness",
        "BMI",
        "Age",
        "Insulin_log",
        "DPF_log",
        "BloodPressure_log",
        "Outcome",
    ]
    clean_df = df[final_cols]
    print("-" * 30)
    print(f"Cleaned data complete: \n{clean_df.head()}")
    return clean_df
