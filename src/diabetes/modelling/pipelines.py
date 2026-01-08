from typing import Sequence

from glum import GeneralizedLinearRegressor
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def glm_pipeline(
    numericals: Sequence[str],
    categoricals: Sequence[str],
) -> Pipeline:
    """
    Build a preprocessing + GLM pipeline for binary classification.

    The pipeline:
    - Mean-imputes numeric features
    - One-hot encodes categorical features (drops first level)
    - Fits a GLM binary classifier

    Parameters
    ----
    numericals (Sequence[str]):
        Names of numeric feature columns.
    categoricals (Sequence[str]):
        Names of categorical feature columns.

    Returns
    ----
    Pipeline:
        A scikit-learn Pipeline with preprocessing and an
        GeneralizedLinearRegressor model.
    """
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numericals,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categoricals,
            ),
        ]
    )

    # Keep the pandas DataFrame output so downstream models retain informative feature names.
    preprocess.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                GeneralizedLinearRegressor(
                    family="binomial",
                    link="logit",
                    max_iter=100,
                ),
            ),
        ]
    )


def lgbm_pipeline(
    numericals: Sequence[str],
    categoricals: Sequence[str],
) -> Pipeline:
    """
    Build a preprocessing + LightGBM pipeline for binary classification.

    The pipeline:
    - Mean-imputes numeric features
    - One-hot encodes categorical features
    - Fits a LightGBM binary classifier

    Parameters
    ----
    numericals (Sequence[str]):
        Names of numeric feature columns.
    categoricals (Sequence[str]):
        Names of categorical feature columns.

    Returns
    ----
    Pipeline:
        A scikit-learn Pipeline with preprocessing and an
        LGBMClassifier model.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numericals),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categoricals,
            ),
        ]
    )

    # Ensure the preprocessor outputs a pandas DataFrame.
    # This ensures feature names are passed to LGBMClassifier, preventing warnings.
    preprocess.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_jobs=1,
                    verbose=-1,
                    force_col_wise=True,
                ),
            ),
        ]
    )
