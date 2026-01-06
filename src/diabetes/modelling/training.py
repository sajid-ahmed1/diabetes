import warnings
from typing import Sequence

from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def glm_search(
    glm_pipe: Pipeline,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    l1_ratios: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    n_jobs: int = 1,
) -> GridSearchCV:
    """
    Construct a GridSearchCV object for tuning a GLM pipeline with elastic-net
    regularization using stratified cross-validation and log-loss scoring.

    Notes
    -----
    - had to add scoring and warnings ignore due to the spam of warnings
    - added make_scorer due to GLM returning probabilities via predict()
    - but the warnings were looking for predict_proba()
    - I used stratified CV because the dataset is imbalanced, it had 5-6% slow
        sessions and the rest not slow. Due to this, I read that using the
        stratified CV handles imbalanced datasets well because when taking the
        fold, it keeps the proportion the same. So the fold would have 5% of
        target = 1 (slow session).

    Parameters
    ----
    glm_pipe (Pipeline):
        scikit-learn pipeline containing a GLM estimator under the step name
        ``"model"``.
    n_splits (int):
        Number of stratified CV folds.
    random_state (int):
        Random seed for shuffling CV splits.
    alphas (Sequence[float]):
        Grid of regularization strengths to search over.
    l1_ratios (Sequence[float]):
        Grid of elastic-net mixing parameters to search over.
    n_jobs (int):
        Number of parallel jobs for grid search.

    Returns
    ----
    GridSearchCV:
        Configured grid search object ready to be fit.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    param_grid = {
        "model__alpha": list(alphas),
        "model__l1_ratio": list(l1_ratios),
    }

    scoring = make_scorer(
        log_loss,
        greater_is_better=False,
        response_method="predict",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Found unknown categories in columns .* during transform.*",
    )

    return GridSearchCV(
        estimator=glm_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )


def lgbm_search(
    lgbm_pipe: Pipeline,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    learning_rates: Sequence[float] = (0.01, 0.03, 0.05, 0.1),
    n_estimators: Sequence[int] = (100, 300, 500),
    num_leaves: Sequence[int] = (15, 31, 63),
    min_child_weight: Sequence[int] = (1, 10, 20),
    n_jobs: int = 1,
) -> GridSearchCV:
    """
    Construct a GridSearchCV object for tuning a LightGBM pipeline using
    cross-validated negative log loss.

    Notes
    -----
    - Assumes a binary classification task.
    - Uses ``neg_log_loss`` so higher scores correspond to better models.
    - Expects the LightGBM estimator to be under the pipeline step name
      ``"model"``.

    Parameters
    ----
    lgbm_pipe (Pipeline):
        scikit-learn pipeline containing a LightGBM model under the step
        name ``"model"``.
    n_splits (int):
        Number of stratified CV folds.
    random_state (int):
        Random seed for shuffling CV splits.
    learning_rates (Sequence[float]):
        Learning rates to search over.
    n_estimators (Sequence[int]):
        Number of boosting iterations to search over.
    num_leaves (Sequence[int]):
        Maximum number of leaves per tree.
    min_child_weight (Sequence[int]):
        Minimum sum of instance weight needed in a child.
    n_jobs (int):
        Number of parallel jobs for grid search.

    Returns
    ----
    GridSearchCV:
        Configured grid search object ready to be fit.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    param_grid = {
        "model__learning_rate": list(learning_rates),
        "model__n_estimators": list(n_estimators),
        "model__num_leaves": list(num_leaves),
        "model__min_child_weight": list(min_child_weight),
    }

    return GridSearchCV(
        estimator=lgbm_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
    )
