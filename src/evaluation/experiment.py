import time

import numpy as np

from src.evaluation.classifier import build_linear_svm
from src.evaluation.metrics import calculate_classification_metrics


def run_feature_experiment(
    extractor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dataset_name: str = "unknown",
    method_name: str = "unknown",
    seed: int | None = None,
) -> dict:
    """
    Run one complete MARS feature-extraction experiment.

    Pipeline
    --------
    1. Fit feature extractor on training data.
    2. Transform training data.
    3. Transform test data.
    4. Fit the common standardised Linear SVM.
    5. Predict training and test labels.
    6. Calculate shared classification metrics.
    7. Return one standardised results dictionary.

    Parameters
    ----------
    extractor:
        Any feature extractor implementing:
            fit(X, y)
            transform(X)

    X_train:
        Training images.

    y_train:
        Training labels.

    X_test:
        Test images.

    y_test:
        Test labels.

    dataset_name:
        Name of the dataset being evaluated.

    method_name:
        Name of the feature-extraction approach.

    seed:
        Experimental random seed, if applicable.

    Returns
    -------
    dict
        Standardised experiment results.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training images and labels must contain "
            "the same number of samples."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test images and labels must contain "
            "the same number of samples."
        )

    if len(X_train) == 0:
        raise ValueError(
            "Training data cannot be empty."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Test data cannot be empty."
        )

    # -------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------

    extraction_start = time.perf_counter()

    extractor.fit(
        X_train,
        y_train,
    )

    train_features = extractor.transform(
        X_train
    )

    test_features = extractor.transform(
        X_test
    )

    extraction_time = (
        time.perf_counter()
        - extraction_start
    )

    train_features = np.asarray(
        train_features,
        dtype=np.float64,
    )

    test_features = np.asarray(
        test_features,
        dtype=np.float64,
    )

    if train_features.ndim != 2:
        raise ValueError(
            "Training feature matrix must be 2D."
        )

    if test_features.ndim != 2:
        raise ValueError(
            "Test feature matrix must be 2D."
        )

    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError(
            "Training and test feature matrices must have "
            "the same number of features."
        )

    if not np.all(np.isfinite(train_features)):
        raise ValueError(
            "Training features contain NaN or infinite values."
        )

    if not np.all(np.isfinite(test_features)):
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    # -------------------------------------------------------
    # Common classifier
    # -------------------------------------------------------

    classifier = build_linear_svm()

    classifier_start = time.perf_counter()

    classifier.fit(
        train_features,
        y_train,
    )

    classifier_time = (
        time.perf_counter()
        - classifier_start
    )

    # -------------------------------------------------------
    # Predictions
    # -------------------------------------------------------

    train_predictions = classifier.predict(
        train_features
    )

    test_predictions = classifier.predict(
        test_features
    )

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------

    train_metrics = (
        calculate_classification_metrics(
            y_train,
            train_predictions,
        )
    )

    test_metrics = (
        calculate_classification_metrics(
            y_test,
            test_predictions,
        )
    )

    # -------------------------------------------------------
    # Standardised result
    # -------------------------------------------------------

    results = {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,

        "n_train": int(
            len(y_train)
        ),

        "n_test": int(
            len(y_test)
        ),

        "n_features": int(
            train_features.shape[1]
        ),

        "feature_extraction_time": float(
            extraction_time
        ),

        "classifier_training_time": float(
            classifier_time
        ),

        "train_metrics": train_metrics,
        "test_metrics": test_metrics,

        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
    }

    # Include extractor-specific information when available.
    if hasattr(
        extractor,
        "train_time_",
    ):
        results["extractor_train_time"] = getattr(
            extractor,
            "train_time_",
        )

    if hasattr(
        extractor,
        "train_fitness_",
    ):
        results["extractor_train_fitness"] = getattr(
            extractor,
            "train_fitness_",
        )

    if hasattr(
        extractor,
        "program_",
    ):
        results["program"] = getattr(
            extractor,
            "program_",
        )

    return results