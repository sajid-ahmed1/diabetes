import pandas as pd


def value_counts(df: pd.DataFrame, column: str) -> None:
    """
    Checks the value counts of the variable with a plot
    """

    print(f"The values within this {column} are : \n{df[column].value_counts()}")
    print("-" * 30)
    df[column].value_counts().plot(kind="bar")
    return None
