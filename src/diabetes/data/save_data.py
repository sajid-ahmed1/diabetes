from pathlib import Path
from typing import Any, Optional, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from sklearn.pipeline import Pipeline

MODEL = Path(__file__).parent.parent.parent.parent / "models"
IMAGES = Path(__file__).parent.parent.parent.parent / "data" / "images"


def save_data(df: pd.DataFrame, save_path: Path) -> Path:
    """
    Save a cleaned DataFrame to a parquet file as required by the coursework.

    Parameters
    ----
    df (pd.DataFrame):
        Cleaned dataset to be saved.
    save_path (Path):
        Directory where the parquet file should be written.

    Returns
    ----
    Path:
        Path to the saved parquet file.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    output_path = save_path / "cleaned_data.parquet"
    df.to_parquet(output_path)

    return output_path


def save_model(pipeline: Pipeline, name: str, **kwargs: Any) -> None:
    """
    Save a trained pipeline as a .pkl file.

    Parameters
    ----
    pipeline (Pipeline):
        taking the trained pipeline
    name (str):
        name of the file
    **kwargs:
        Additional arguments passed to joblib.dump (e.g. compress=3).
    """
    MODEL.mkdir(parents=True, exist_ok=True)

    # remove extension
    base = name.split(".")[0]

    # replace unsafe characters
    clean = "".join(c if c.isalnum() or c in "-_ " else "_" for c in base)

    joblib.dump(pipeline, MODEL / (clean + ".pkl"), **kwargs)


def save_image(
    obj: Optional[Union[Figure, Axes]] = None,
    filename: Union[str, Path] = "",
    dpi: int = 300,
    tight: bool = True,
    transparent: bool = False,
) -> Path:
    """
    Save a Matplotlib Figure or Axes to the IMAGES directory.

    Parameters
    ----
    obj (Figure | Axes | None):
        figure, axes, or None to use the current active figure.
    filename (str | Path):
        relative filename inside the images directory.
    dpi (int):
        dots per inch used when saving.
    tight (bool):
        whether to use a tight bounding box.
    transparent (bool):
        whether to save with a transparent background.

    Notes
    ----
    - For plots that do not return a Figure/Axes (e.g. SHAP beeswarm),
      pass obj=None and the current active figure (plt.gcf()) is saved.

    Returns
    ----
    Path:
        Path to the saved image file.
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / Path(filename)

    # Normalize to a Matplotlib figure-like object
    fig: Figure | SubFigure
    if obj is None:
        fig = plt.gcf()
    elif isinstance(obj, Axes):
        fig = obj.figure
    elif isinstance(obj, (Figure, SubFigure)):
        fig = obj
    else:
        raise TypeError("Expected Figure, Axes, or None")

    kwargs: dict[str, Any] = {"dpi": dpi, "transparent": transparent}
    if tight:
        kwargs["bbox_inches"] = "tight"

    # SubFigure does not expose savefig in matplotlib stubs; save via the parent Figure.
    save_fig: Figure = fig.figure if isinstance(fig, SubFigure) else fig

    save_fig.savefig(path, **kwargs)
    return path
