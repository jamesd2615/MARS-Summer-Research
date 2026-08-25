import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculate the standard classification metrics used
    throughout the MARS experiments.

    Metrics returned for all classification tasks:
        - accuracy
        - balanced_accuracy
        - macro_f1
        - weighted_f1

    For binary classification tasks, the function also returns:
        - precision
        - recall

    Parameters
    ----------
    y_true:
        True class labels.

    y_pred:
        Predicted class labels.

    Returns
    -------
    dict
        Dictionary containing classification metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be a one-dimensional array."
        )

    if y_pred.ndim != 1:
        raise ValueError(
            "y_pred must be a one-dimensional array."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same "
            "number of samples."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics for empty arrays."
        )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    unique_labels = np.unique(
        y_true
    )

    if len(unique_labels) == 2:
        positive_label = unique_labels[-1]

        metrics["precision"] = float(
            precision_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
        )

        metrics["recall"] = float(
            recall_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
        )

    return metrics