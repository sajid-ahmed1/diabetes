import os
from pathlib import Path
from typing import cast

import kagglehub
import pandas as pd
from dotenv import load_dotenv
from kagglehub import KaggleDatasetAdapter

project_data_dir = (
    Path.cwd() / "data" / "raw"
)  # this would be useful when we have the csv download
file_path = "Healthcare-Diabetes.csv"

# load credentials from .env file
load_dotenv()

# set environment variables for kagglehub/kaggle API
os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN", "")


def read_data() -> pd.DataFrame:
    """
    Pulls the Kaggle dataset via API and returns into Pandas dataframe

    :return: Description
    """
    # load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nanditapore/healthcare-diabetes",
        file_path,
    )
    print("-" * 30)
    print(f"Data ingestion complete: \n{df.head()}")
    return cast(pd.DataFrame, df)
