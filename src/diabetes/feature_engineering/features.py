import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding feature engineered columns from analysis
    """
    df["Glucose_Insulin_Interaction"] = df["Glucose"] * df["Insulin_log"]
    df["Age_Insulin_Interaction"] = df["Age"] * df["Insulin_log"]
    print("-" * 30)
    print(f"Features added complete: \n{df.head()}")
    return df
