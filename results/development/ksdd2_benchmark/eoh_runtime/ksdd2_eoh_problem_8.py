
from __future__ import annotations

import time
import traceback

import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eoh import BaseProblem


class KSDD2FeatureExtractionProblem8(BaseProblem):
    """
    EOH problem for KSDD2 binary defect classification.

    Every valid candidate must return exactly eight finite
    scalar features from one 64 x 64 grayscale image.
    """

    N_FEATURES = 8

    template_program = """
import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    mean_value = np.mean(image)
    std_value = np.std(image)

    dark_threshold = mean_value - std_value
    bright_threshold = mean_value + std_value

    laplacian_like = (
        -4.0 * image[1:-1, 1:-1]
        + image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
    )

    return np.array([
        mean_value,
        std_value,
        np.mean(np.abs(dx)),
        np.mean(np.abs(dy)),
        np.std(
            np.concatenate([
                dx.reshape(-1),
                dy.reshape(-1),
            ])
        ),
        np.mean(image < dark_threshold),
        np.mean(image > bright_threshold),
        np.var(laplacian_like),
    ], dtype=float)
"""


    task_description = (
        "Improve the NumPy-only extract_features function for binary "
        "industrial surface-defect classification on the KolektorSDD2 "
        "dataset. The input is one preprocessed 64 by 64 grayscale NumPy "
        "array with values between 0 and 1. The function must be named "
        "extract_features, accept exactly one image argument, and return "
        "exactly eight finite scalar values in a one-dimensional NumPy "
        "array with shape (8,). Each feature should have a stable and "
        "interpretable image-processing meaning. Use only deterministic "
        "NumPy computation. Do not access files, networks, random number "
        "generators, classifiers, labels, masks, global datasets or "
        "external libraries. Prefer interpretable measurements relevant "
        "to surface defects, including intensity statistics, gradients, "
        "local contrast, edge strength, texture variation, bright or dark "
        "pixel proportions, regional statistics and simple moments. "
        "Do not return flattened images, long histograms, arbitrary pixel "
        "collections or high-dimensional vectors. Candidate quality is "
        "measured using balanced validation accuracy from a fixed "
        "standardised class-balanced linear SVM. Fitness equals one minus "
        "balanced validation accuracy, so lower fitness is better."
    )


    def __init__(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        *,
        timeout=180,
        n_processes=1,
        max_seconds_per_evaluation=300.0,
    ):
        super().__init__(
            timeout=timeout,
            n_processes=n_processes,
        )

        self.X_train = np.asarray(
            X_train,
            dtype=np.float32,
        )

        self.y_train = np.asarray(
            y_train,
            dtype=np.int64,
        )

        self.X_validation = np.asarray(
            X_validation,
            dtype=np.float32,
        )

        self.y_validation = np.asarray(
            y_validation,
            dtype=np.int64,
        )

        self.max_seconds_per_evaluation = float(
            max_seconds_per_evaluation
        )


    @classmethod
    def _validate_vector(
        cls,
        values,
        image_index,
    ):
        vector = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if vector.shape != (cls.N_FEATURES,):
            raise ValueError(
                f"Image {image_index} produced "
                f"{vector.shape}; exactly "
                f"({cls.N_FEATURES},) is required."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Image {image_index} produced "
                "NaN or infinite values."
            )

        return vector


    def _build_feature_matrix(
        self,
        images,
        feature_function,
    ):
        rows = []

        started = time.perf_counter()

        for image_index, image in enumerate(images):

            if (
                time.perf_counter() - started
                > self.max_seconds_per_evaluation
            ):
                raise TimeoutError(
                    "Feature extraction exceeded "
                    "the local evaluation limit."
                )

            rows.append(
                self._validate_vector(
                    feature_function(image),
                    image_index,
                )
            )

        matrix = np.vstack(rows)

        expected_shape = (
            len(images),
            self.N_FEATURES,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Expected feature matrix "
                f"{expected_shape}, got "
                f"{matrix.shape}."
            )

        return matrix


    def _validation_balanced_accuracy(
        self,
        feature_function,
    ):
        training_features = (
            self._build_feature_matrix(
                self.X_train,
                feature_function,
            )
        )

        validation_features = (
            self._build_feature_matrix(
                self.X_validation,
                feature_function,
            )
        )

        constant_columns = np.where(
            np.isclose(
                training_features.std(axis=0),
                0.0,
            )
        )[0]

        if len(constant_columns) > 0:
            raise ValueError(
                "Constant training features in "
                f"columns {constant_columns.tolist()}."
            )

        scaler = StandardScaler()

        training_scaled = scaler.fit_transform(
            training_features
        )

        validation_scaled = scaler.transform(
            validation_features
        )

        classifier = SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )

        classifier.fit(
            training_scaled,
            self.y_train,
        )

        predictions = classifier.predict(
            validation_scaled
        )

        return float(
            balanced_accuracy_score(
                self.y_validation,
                predictions,
            )
        )


    def evaluate_program(
        self,
        program_str,
        callable_func,
    ):
        started = time.perf_counter()

        try:
            if callable_func is None:
                raise TypeError(
                    "EOH did not provide a "
                    "callable function."
                )

            self._validate_vector(
                callable_func(
                    self.X_train[0]
                ),
                0,
            )

            balanced_accuracy = (
                self._validation_balanced_accuracy(
                    callable_func
                )
            )

            if not (
                0.0
                <= balanced_accuracy
                <= 1.0
            ):
                raise ValueError(
                    "Balanced validation accuracy "
                    "is outside [0, 1]."
                )

            fitness = float(
                1.0 - balanced_accuracy
            )

            self._write_diagnostic(
                "success",
                program_str,
                (
                    "feature_dim=8, "
                    f"balanced_accuracy="
                    f"{balanced_accuracy:.6f}, "
                    f"fitness={fitness:.6f}, "
                    f"runtime="
                    f"{time.perf_counter() - started:.3f}s"
                ),
            )

            return fitness

        except Exception as error:

            self._write_diagnostic(
                "failure",
                program_str,
                (
                    f"{type(error).__name__}: "
                    f"{error}\n"
                    f"{traceback.format_exc()}"
                ),
            )

            return None


    def _write_diagnostic(
        self,
        status,
        program_str,
        message,
    ):
        try:
            from datetime import datetime
            from pathlib import Path
            import os

            folder = (
                Path(__file__).resolve().parents[2]
                / "ksdd2_eoh_8"
                / "candidate_diagnostics"
            )

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            path = (
                folder
                / f"{stamp}_{os.getpid()}_"
                  f"{status}.txt"
            )

            path.write_text(
                (
                    f"STATUS: {status}\n\n"
                    f"MESSAGE:\n{message}\n\n"
                    f"PROGRAM:\n{program_str}\n"
                ),
                encoding="utf-8",
            )

        except Exception:
            pass
