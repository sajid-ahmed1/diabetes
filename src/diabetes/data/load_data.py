from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

# define the paths
ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "data" / "raw" / "Healthcare-Diabetes.csv"
PARQUET_PATH = ROOT / "data" / "cleaned_data.parquet"
MODEL = ROOT / "models"
IMAGES = ROOT / "images"


def load_csv() -> pd.DataFrame:
    """
    Loads the csv data into a pandas dataframe

    Returns
    ----
    pd.DataFrame:
        The loaded dataset.
    """

    # load the dataset
    df = pd.read_csv(CSV_PATH)
    # copy the dataset to avoid altering the original data
    # (crucial) because I am using a direct csv import
    df_copy = df.copy()
    return df_copy


def load_parquet() -> pd.DataFrame:
    """
    Load a Parquet dataset into a pandas DataFrame.

    Returns
    ----
    pd.DataFrame:
        The loaded dataset.
    """
    df = pd.read_parquet(PARQUET_PATH)
    return df.copy()


def load_model(name: str) -> Pipeline:
    """
    Loads a model from the models directory.

    Parameters
    ----
    name (str):
        name of the model

    Returns
    ----
    Pipeline:
        model pipeline
    """
    return joblib.load(MODEL / f"{name}.pkl")


def save_image(fig: Figure | None, name: str) -> None:
    """
    Saves a matplotlib figure to the images directory.

    Parameters
    ----
    fig (Figure | None):
        The figure to save. If None, saves the current active figure.
    name (str):
        The filename (e.g. "plot.png").
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / name
    if fig is None:
        plt.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")
    print(f"Saved image to {path}")
