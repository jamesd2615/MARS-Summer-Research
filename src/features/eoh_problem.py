from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from eoh import BaseProblem

from src.config import RESULTS_DIR


class EOHFeatureExtractionProblem(BaseProblem):
    """
    EOH problem for explainable image feature extraction.

    Candidate programs must return exactly eight finite,
    interpretable scalar features for each image.

    Fitness is:

        1 - validation accuracy

    using a fixed standardised Linear SVM.
    """

    N_FEATURES = 8

    template_program = """
import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    centre = image[
        int(0.20 * h):int(0.85 * h),
        int(0.20 * w):int(0.80 * w)
    ]

    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    half = w // 2

    left = image[:, :half]
    right = np.fliplr(image[:, w-half:])

    threshold = np.mean(image) + np.std(image)

    return np.array(
        [
            np.mean(image),
            np.std(image),
            np.mean(centre),
            np.std(centre),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(np.abs(left - right)),
            np.mean(image > threshold),
        ],
        dtype=float,
    )
"""

    task_description = (
        "Improve the NumPy-only extract_features function for image "
        "classification. The input is one grayscale NumPy image array. "
        "The function must be named extract_features, accept exactly one "
        "image, and return exactly eight finite scalar values in a "
        "one-dimensional NumPy array with shape (8,). Each feature should "
        "have a stable and concise image-processing interpretation. Use only "
        "deterministic NumPy computation. Do not access files, networks, "
        "random generators, classifiers, labels, global datasets or external "
        "libraries. Prefer regional intensity statistics, symmetry, gradients, "
        "contrasts, proportions, moments and simple texture measurements. "
        "Do not return flattened images, long histograms, arbitrary pixel "
        "collections or high-dimensional vectors. Fitness is one minus "
        "validation accuracy from a fixed standardised Linear SVM, so lower "
        "fitness is better."
    )

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        timeout: int = 180,
        n_processes: int = 1,
        max_seconds_per_evaluation: float = 150.0,
        diagnostic_subdir: str = "eoh",
    ):
        super().__init__(
            timeout=timeout,
            n_processes=n_processes,
        )

        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

        self.X_validation = np.asarray(
            X_validation
        )

        self.y_validation = np.asarray(
            y_validation
        )

        self.max_seconds_per_evaluation = float(
            max_seconds_per_evaluation
        )

        self.diagnostic_dir = (
            RESULTS_DIR
            / diagnostic_subdir
            / "candidate_diagnostics"
        )

    @classmethod
    def _validate_vector(
        cls,
        values,
        image_index: int,
    ) -> np.ndarray:
        """
        Validate one EOH-generated feature vector.
        """
        vector = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if vector.shape != (
            cls.N_FEATURES,
        ):
            raise ValueError(
                f"Image {image_index} produced "
                f"{vector.shape}; exactly "
                f"({cls.N_FEATURES},) is required."
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                f"Image {image_index} produced "
                "NaN or infinite values."
            )

        return vector

    def _build_feature_matrix(
        self,
        images: np.ndarray,
        feature_function,
    ) -> np.ndarray:
        """
        Apply one candidate feature function to a collection
        of images.
        """
        rows = []

        started = time.perf_counter()

        for image_index, image in enumerate(
            images
        ):
            elapsed = (
                time.perf_counter()
                - started
            )

            if (
                elapsed
                > self.max_seconds_per_evaluation
            ):
                raise TimeoutError(
                    "Feature extraction exceeded the "
                    "local evaluation limit."
                )

            feature_vector = (
                feature_function(image)
            )

            rows.append(
                self._validate_vector(
                    feature_vector,
                    image_index,
                )
            )

        matrix = np.vstack(
            rows
        )

        expected_shape = (
            len(images),
            self.N_FEATURES,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Expected feature matrix "
                f"{expected_shape}, "
                f"received {matrix.shape}."
            )

        return matrix

    def _validation_accuracy(
        self,
        feature_function,
    ) -> float:
        """
        Evaluate one candidate feature extractor using
        training and validation data only.
        """
        train_features = (
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
                train_features.std(axis=0),
                0.0,
            )
        )[0]

        if len(constant_columns) > 0:
            raise ValueError(
                "Constant training features found "
                f"in columns "
                f"{constant_columns.tolist()}."
            )

        classifier = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                random_state=42,
                dual="auto",
                max_iter=10000,
            ),
        )

        classifier.fit(
            train_features,
            self.y_train,
        )

        predictions = classifier.predict(
            validation_features
        )

        return float(
            accuracy_score(
                self.y_validation,
                predictions,
            )
        )

    def evaluate_program(
        self,
        program_str: str,
        callable_func,
    ) -> float | None:
        """
        Evaluate one EOH-generated candidate program.
        """
        started = time.perf_counter()

        try:
            if callable_func is None:
                raise TypeError(
                    "EOH did not provide a "
                    "callable feature function."
                )

            self._validate_vector(
                callable_func(
                    self.X_train[0]
                ),
                0,
            )

            validation_accuracy = (
                self._validation_accuracy(
                    callable_func
                )
            )

            if not (
                0.0
                <= validation_accuracy
                <= 1.0
            ):
                raise ValueError(
                    "Validation accuracy is "
                    "outside [0, 1]."
                )

            fitness = float(
                1.0
                - validation_accuracy
            )

            self._write_diagnostic(
                status="success",
                program_str=program_str,
                message=(
                    f"feature_dim="
                    f"{self.N_FEATURES}, "
                    f"accuracy="
                    f"{validation_accuracy:.6f}, "
                    f"fitness="
                    f"{fitness:.6f}, "
                    f"runtime="
                    f"{time.perf_counter() - started:.3f}s"
                ),
            )

            return fitness

        except Exception as error:
            self._write_diagnostic(
                status="failure",
                program_str=program_str,
                message=(
                    f"{type(error).__name__}: "
                    f"{error}\n"
                    f"{traceback.format_exc()}"
                ),
            )

            return None

    def _write_diagnostic(
        self,
        status: str,
        program_str: str,
        message: str,
    ) -> None:
        """
        Save diagnostic information for EOH candidate
        programs without interrupting the search.
        """
        try:
            from datetime import datetime
            import os

            self.diagnostic_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            path = (
                self.diagnostic_dir
                / (
                    f"{stamp}_"
                    f"{os.getpid()}_"
                    f"{status}.txt"
                )
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