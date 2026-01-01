import pandas as pd


def health_check(df: pd.DataFrame) -> None:
    """
    Checks the dataset for data types, nulls and duplicated values.

    Returns: print statement
    """

    print(f"Total null fields per column:\n{df.isnull().sum()}")
    print("-" * 30)
    print(f"Total duplicated rows: {df.duplicated().sum()}")
    print("-" * 30)
    print(f"Here are the data types:\n{df.dtypes}")
    return None
