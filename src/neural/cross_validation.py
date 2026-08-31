from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch import nn

from src.neural.training import run_neural_experiment


def run_neural_cross_validation(
    model_builder: Callable[[], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    *,
    dataset_name: str,
    method_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> pd.DataFrame:
    """
    Run stratified cross-validation for a neural baseline.

    A fresh model is created for every fold using ``model_builder``.

    Parameters
    ----------
    model_builder:
        Callable that returns a newly initialised PyTorch model.

    X:
        Image array.

    y:
        Target labels.

    dataset_name:
        Dataset identifier.

    method_name:
        Neural method identifier, for example ``cnn`` or ``mlp``.

    n_splits:
        Number of stratified cross-validation folds.

    random_state:
        Base random seed.

    epochs:
        Training epochs per fold.

    batch_size:
        Training and prediction batch size.

    learning_rate:
        Adam learning rate.

    Returns
    -------
    pandas.DataFrame
        One row per cross-validation fold.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(
            "Labels must be a 1D array."
        )

    if len(X) != len(y):
        raise ValueError(
            "Images and labels must contain "
            "the same number of samples."
        )

    if len(X) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    unique_labels, class_counts = np.unique(
        y,
        return_counts=True,
    )

    if len(unique_labels) < 2:
        raise ValueError(
            "Cross-validation requires at least two classes."
        )

    if np.min(class_counts) < n_splits:
        raise ValueError(
            "Every class must contain at least n_splits samples."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows: list[dict] = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        fold_seed = (
            random_state
            + fold_number
            - 1
        )

        print(
            f"\nStarting neural fold "
            f"{fold_number}/{n_splits}"
        )

        print(
            f"Fold seed: {fold_seed}"
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

        # A completely new model is required for every fold.
        model = model_builder()

        result = run_neural_experiment(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            dataset_name=dataset_name,
            method_name=method_name,
            seed=fold_seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        row = {
            "dataset": dataset_name,
            "method": method_name,
            "fold": int(
                fold_number
            ),
            "seed": int(
                fold_seed
            ),

            "n_train": int(
                result["n_train"]
            ),

            "n_validation": int(
                result["n_test"]
            ),

            "epochs": int(
                result["epochs"]
            ),

            "batch_size": int(
                result["batch_size"]
            ),

            "learning_rate": float(
                result["learning_rate"]
            ),

            "model_training_time": float(
                result["model_training_time"]
            ),

            "device": result[
                "device"
            ],

            "initial_training_loss": float(
                result["initial_training_loss"]
            ),
            "final_training_loss": float(
                result["final_training_loss"]
            ),
            "training_loss_change": float(
                result["training_loss_change"]
            ),
        }

        # ---------------------------------------------------
        # Shared train metrics
        # ---------------------------------------------------

        for metric_name, metric_value in result[
            "train_metrics"
        ].items():
            row[
                f"train_{metric_name}"
            ] = float(
                metric_value
            )

        # ---------------------------------------------------
        # Shared validation metrics
        # ---------------------------------------------------

        for metric_name, metric_value in result[
            "test_metrics"
        ].items():
            row[
                f"validation_{metric_name}"
            ] = float(
                metric_value
            )

        fold_rows.append(
            row
        )

        print(
            "Training loss: "
            f"{row['initial_training_loss']:.6f}"
            " -> "
            f"{row['final_training_loss']:.6f}"
        )

        print(
            "Validation accuracy: "
            f"{row['validation_accuracy']:.6f}"
        )

        print(
            "Validation balanced accuracy: "
            f"{row['validation_balanced_accuracy']:.6f}"
        )

        print(
            "Validation macro F1: "
            f"{row['validation_macro_f1']:.6f}"
        )

    return pd.DataFrame(
        fold_rows
    )


def summarise_neural_cross_validation(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise neural cross-validation results.

    Numeric columns beginning with ``train_`` or ``validation_``
    are reported using mean and sample standard deviation.
    """

    if fold_results.empty:
        raise ValueError(
            "fold_results cannot be empty."
        )

    metric_columns = [
        column
        for column in fold_results.columns
        if (
            column.startswith("train_")
            or column.startswith("validation_")
        )
        and pd.api.types.is_numeric_dtype(
            fold_results[column]
        )
    ]

    summary_rows: list[dict] = []

    for column in metric_columns:
        values = fold_results[
            column
        ].astype(float)

        summary_rows.append(
            {
                "metric": column,
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std(
                        ddof=1
                    )
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )