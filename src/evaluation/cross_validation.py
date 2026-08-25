from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation.experiment import run_feature_experiment


def run_stratified_cross_validation(
    extractor,
    X: np.ndarray,
    y: np.ndarray,
    *,
    dataset_name: str,
    method_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified cross-validation for one feature extraction method.

    Each fold:
        1. creates train/validation indices;
        2. fits a fresh feature extractor on the training fold;
        3. transforms train and validation images;
        4. fits the common Linear SVM;
        5. calculates standard metrics;
        6. records one row of fold-level results.

    Parameters
    ----------
    extractor:
        A configured feature extractor implementing:
            fit(X, y)
            transform(X)

    X:
        Image array.

    y:
        Class labels.

    dataset_name:
        Dataset identifier stored in the results.

    method_name:
        Feature-extraction method identifier.

    n_splits:
        Number of stratified folds.

    random_state:
        Seed used to construct the folds.

    Returns
    -------
    pandas.DataFrame
        One row per fold.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if len(X) == 0:
        raise ValueError(
            "Cross-validation data cannot be empty."
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    _, class_counts = np.unique(
        y,
        return_counts=True,
    )

    if np.min(class_counts) < n_splits:
        raise ValueError(
            "Every class must contain at least n_splits samples. "
            f"Smallest class has {np.min(class_counts)} samples, "
            f"but n_splits={n_splits}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        print(
            f"Running fold {fold_number}/{n_splits} "
            f"for {dataset_name} - {method_name}"
        )

        X_train = X[
            train_indices
        ]

        y_train = y[
            train_indices
        ]

        X_validation = X[
            validation_indices
        ]

        y_validation = y[
            validation_indices
        ]

        # Every fold must receive a fresh extractor.
        # This prevents information learned in one fold from
        # leaking into another fold.
        fold_extractor = deepcopy(
            extractor
        )

        # If the extractor exposes a seed, give each fold a
        # deterministic but distinct seed.
        fold_seed = (
            random_state
            + fold_number
            - 1
        )

        if hasattr(
            fold_extractor,
            "seed",
        ):
            fold_extractor.seed = (
                fold_seed
            )

        results = run_feature_experiment(
            fold_extractor,
            X_train,
            y_train,
            X_validation,
            y_validation,
            dataset_name=dataset_name,
            method_name=method_name,
            seed=fold_seed,
        )

        row = {
            "dataset": dataset_name,
            "method": method_name,
            "fold": fold_number,
            "seed": fold_seed,

            "n_train": results[
                "n_train"
            ],

            "n_validation": results[
                "n_test"
            ],

            "n_features": results[
                "n_features"
            ],

            "feature_extraction_time": results[
                "feature_extraction_time"
            ],

            "classifier_training_time": results[
                "classifier_training_time"
            ],

            "train_accuracy": results[
                "train_metrics"
            ]["accuracy"],

            "train_balanced_accuracy": results[
                "train_metrics"
            ]["balanced_accuracy"],

            "train_macro_f1": results[
                "train_metrics"
            ]["macro_f1"],

            "train_weighted_f1": results[
                "train_metrics"
            ]["weighted_f1"],

            "validation_accuracy": results[
                "test_metrics"
            ]["accuracy"],

            "validation_balanced_accuracy": results[
                "test_metrics"
            ]["balanced_accuracy"],

            "validation_macro_f1": results[
                "test_metrics"
            ]["macro_f1"],

            "validation_weighted_f1": results[
                "test_metrics"
            ]["weighted_f1"],
        }

        if (
            "precision"
            in results["train_metrics"]
        ):
            row[
                "train_precision"
            ] = results[
                "train_metrics"
            ]["precision"]

            row[
                "train_recall"
            ] = results[
                "train_metrics"
            ]["recall"]

        if (
            "precision"
            in results["test_metrics"]
        ):
            row[
                "validation_precision"
            ] = results[
                "test_metrics"
            ]["precision"]

            row[
                "validation_recall"
            ] = results[
                "test_metrics"
            ]["recall"]

        if (
            results.get(
                "extractor_train_time"
            )
            is not None
        ):
            row[
                "extractor_train_time"
            ] = results[
                "extractor_train_time"
            ]

        if (
            results.get(
                "extractor_train_fitness"
            )
            is not None
        ):
            row[
                "extractor_train_fitness"
            ] = results[
                "extractor_train_fitness"
            ]

        if (
            results.get(
                "program"
            )
            is not None
        ):
            row[
                "program"
            ] = results[
                "program"
            ]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarise_cross_validation(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise fold-level cross-validation results.

    Numeric metrics are reported as mean and standard deviation.
    """
    if fold_results.empty:
        raise ValueError(
            "Cannot summarise an empty cross-validation table."
        )

    metric_columns = [
        column
        for column in fold_results.columns
        if (
            column.startswith(
                "train_"
            )
            or column.startswith(
                "validation_"
            )
        )
        and column not in {
            "train_predictions",
            "validation_predictions",
        }
        and np.issubdtype(
            fold_results[
                column
            ].dtype,
            np.number,
        )
    ]

    rows = []

    for metric in metric_columns:
        values = fold_results[
            metric
        ].dropna()

        rows.append(
            {
                "metric": metric,
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std(
                        ddof=1
                    )
                ),
                "n_folds": int(
                    len(values)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )