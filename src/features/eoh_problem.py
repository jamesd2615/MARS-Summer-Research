from __future__ import annotations

import time
import traceback

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

    Supports grayscale images with shape (n, h, w) and RGB images
    with shape (n, h, w, 3). Candidate programs must return exactly
    eight finite, interpretable scalar features per image.

    Fitness = 1 - validation accuracy using a fixed standardised Linear SVM.
    """

    N_FEATURES = 8

    GRAYSCALE_TEMPLATE_PROGRAM = """
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

    RGB_TEMPLATE_PROGRAM = """
import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")

    h, w, _ = image.shape

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    intensity = np.mean(image, axis=2)

    centre = intensity[
        int(0.20 * h):int(0.85 * h),
        int(0.20 * w):int(0.80 * w)
    ]

    dx = np.diff(intensity, axis=1)
    dy = np.diff(intensity, axis=0)
    colour_range = np.max(image, axis=2) - np.min(image, axis=2)

    return np.array(
        [
            np.mean(red),
            np.mean(green),
            np.mean(blue),
            np.std(intensity),
            np.mean(centre) - np.mean(intensity),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(colour_range),
        ],
        dtype=float,
    )
"""

    GRAYSCALE_TASK_DESCRIPTION = (
        "Improve the NumPy-only extract_features function for image "
        "classification. The input is one grayscale NumPy image array with "
        "shape (height, width). The function must be named extract_features, "
        "accept exactly one image, and return exactly eight finite scalar "
        "values in a one-dimensional NumPy array with shape (8,). Each "
        "feature should have a stable and concise image-processing "
        "interpretation. Use only deterministic NumPy computation. Do not "
        "access files, networks, random generators, classifiers, labels, "
        "global datasets or external libraries. Prefer regional intensity "
        "statistics, symmetry, gradients, contrasts, proportions, moments "
        "and simple texture measurements. Do not return flattened images, "
        "long histograms, arbitrary pixel collections or high-dimensional "
        "vectors. Fitness is one minus validation accuracy from a fixed "
        "standardised Linear SVM, so lower fitness is better."
    )

    RGB_TASK_DESCRIPTION = (
        "Improve the NumPy-only extract_features function for RGB image "
        "classification. The input is one RGB NumPy image array with shape "
        "(height, width, 3), where the final axis contains red, green and "
        "blue channels. The function must be named extract_features, accept "
        "exactly one image, and return exactly eight finite scalar values in "
        "a one-dimensional NumPy array with shape (8,). Each feature should "
        "have a stable, concise and explainable image-processing "
        "interpretation. Use only deterministic NumPy computation. Do not "
        "access files, networks, random generators, classifiers, labels, "
        "global datasets or external libraries. Colour information may be "
        "used explicitly, including channel statistics, channel contrasts, "
        "colour spread or simple colour relationships. Spatial information "
        "may also be used through regional statistics, symmetry, gradients, "
        "contrasts, proportions and simple texture measurements. Do not "
        "return flattened images, long histograms, arbitrary pixel "
        "collections or high-dimensional vectors. Fitness is one minus "
        "validation accuracy from a fixed standardised Linear SVM, so lower "
        "fitness is better."
    )

    template_program = GRAYSCALE_TEMPLATE_PROGRAM
    task_description = GRAYSCALE_TASK_DESCRIPTION

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
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_validation = np.asarray(X_validation)
        self.y_validation = np.asarray(y_validation)

        if self.X_train.ndim == 3:
            self.input_mode = "grayscale"
            self.template_program = self.GRAYSCALE_TEMPLATE_PROGRAM
            self.task_description = self.GRAYSCALE_TASK_DESCRIPTION
        elif self.X_train.ndim == 4 and self.X_train.shape[-1] == 3:
            self.input_mode = "rgb"
            self.template_program = self.RGB_TEMPLATE_PROGRAM
            self.task_description = self.RGB_TASK_DESCRIPTION
        else:
            raise ValueError(
                "EOHFeatureExtractionProblem expects grayscale image data "
                "with shape (n_samples, height, width) or RGB image data "
                "with shape (n_samples, height, width, 3). "
                f"Received training shape {self.X_train.shape}."
            )

        self._validate_dataset_shapes()

        super().__init__(timeout=timeout, n_processes=n_processes)

        self.max_seconds_per_evaluation = float(max_seconds_per_evaluation)
        self.diagnostic_dir = RESULTS_DIR / diagnostic_subdir / "candidate_diagnostics"

    def _validate_dataset_shapes(self) -> None:
        if len(self.X_train) != len(self.y_train):
            raise ValueError("Training images and labels must have equal length.")
        if len(self.X_validation) != len(self.y_validation):
            raise ValueError("Validation images and labels must have equal length.")
        if len(self.X_train) == 0 or len(self.X_validation) == 0:
            raise ValueError("Training and validation sets must be non-empty.")

        if self.input_mode == "grayscale" and self.X_validation.ndim != 3:
            raise ValueError(
                "EOH problem was initialised for grayscale data, but "
                f"validation shape is {self.X_validation.shape}."
            )
        if self.input_mode == "rgb" and (
            self.X_validation.ndim != 4 or self.X_validation.shape[-1] != 3
        ):
            raise ValueError(
                "EOH problem was initialised for RGB data, but "
                f"validation shape is {self.X_validation.shape}."
            )

    def _validate_image_shape(self, image: np.ndarray, image_index: int) -> np.ndarray:
        image = np.asarray(image)
        if self.input_mode == "grayscale" and image.ndim != 2:
            raise ValueError(
                f"Image {image_index} has shape {image.shape}; "
                "expected grayscale shape (height, width)."
            )
        if self.input_mode == "rgb" and (image.ndim != 3 or image.shape[-1] != 3):
            raise ValueError(
                f"Image {image_index} has shape {image.shape}; "
                "expected RGB shape (height, width, 3)."
            )
        return image

    @classmethod
    def _validate_vector(cls, values, image_index: int) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (cls.N_FEATURES,):
            raise ValueError(
                f"Image {image_index} produced {vector.shape}; exactly "
                f"({cls.N_FEATURES},) is required."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Image {image_index} produced NaN or infinite values.")
        return vector

    def _build_feature_matrix(self, images: np.ndarray, feature_function) -> np.ndarray:
        rows = []
        started = time.perf_counter()

        for image_index, image in enumerate(images):
            if time.perf_counter() - started > self.max_seconds_per_evaluation:
                raise TimeoutError("Feature extraction exceeded the local evaluation limit.")

            image = self._validate_image_shape(image, image_index)
            feature_vector = feature_function(image)
            rows.append(self._validate_vector(feature_vector, image_index))

        matrix = np.vstack(rows)
        expected_shape = (len(images), self.N_FEATURES)
        if matrix.shape != expected_shape:
            raise ValueError(f"Expected feature matrix {expected_shape}, received {matrix.shape}.")
        return matrix

    def _validation_accuracy(self, feature_function) -> float:
        train_features = self._build_feature_matrix(self.X_train, feature_function)
        validation_features = self._build_feature_matrix(
            self.X_validation, feature_function
        )

        constant_columns = np.where(
            np.isclose(train_features.std(axis=0), 0.0)
        )[0]
        if len(constant_columns) > 0:
            raise ValueError(
                "Constant training features found in columns "
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
        classifier.fit(train_features, self.y_train)
        predictions = classifier.predict(validation_features)
        return float(accuracy_score(self.y_validation, predictions))

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        started = time.perf_counter()
        try:
            if callable_func is None:
                raise TypeError("EOH did not provide a callable feature function.")

            first_image = self._validate_image_shape(self.X_train[0], 0)
            self._validate_vector(callable_func(first_image), 0)

            validation_accuracy = self._validation_accuracy(callable_func)
            if not 0.0 <= validation_accuracy <= 1.0:
                raise ValueError("Validation accuracy is outside [0, 1].")

            fitness = float(1.0 - validation_accuracy)
            self._write_diagnostic(
                status="success",
                program_str=program_str,
                message=(
                    f"input_mode={self.input_mode}, "
                    f"feature_dim={self.N_FEATURES}, "
                    f"accuracy={validation_accuracy:.6f}, "
                    f"fitness={fitness:.6f}, "
                    f"runtime={time.perf_counter() - started:.3f}s"
                ),
            )
            return fitness

        except Exception as error:
            self._write_diagnostic(
                status="failure",
                program_str=program_str,
                message=(
                    f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                ),
            )
            return None

    def _write_diagnostic(self, status: str, program_str: str, message: str) -> None:
        try:
            from datetime import datetime
            import os

            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = self.diagnostic_dir / f"{stamp}_{os.getpid()}_{status}.txt"
            path.write_text(
                f"STATUS: {status}\n\nMESSAGE:\n{message}\n\nPROGRAM:\n{program_str}\n",
                encoding="utf-8",
            )
        except Exception:
            pass
