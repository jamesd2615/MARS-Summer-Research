from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.base import BaseFeatureExtractor


class EOHExtractor(BaseFeatureExtractor):
    """
    Modular wrapper for an EOH-generated feature extraction program.

    The expensive EOH/LLM search is performed separately using
    EOHFeatureExtractionProblem.

    Once EOH has selected a program, this class:

        1. stores the selected program text;
        2. compiles the program;
        3. retrieves its extract_features() function;
        4. validates the feature contract;
        5. applies the program to arbitrary image arrays.

    The selected EOH program must define:

        def extract_features(image):
            ...

    and must return exactly `n_features` finite scalar values.
    """

    def __init__(
        self,
        program_text: str | None = None,
        *,
        n_features: int = 8,
    ):
        self.program_text = (
            program_text.strip()
            if program_text is not None
            else None
        )

        self.n_features = int(
            n_features
        )

        if self.n_features <= 0:
            raise ValueError(
                "n_features must be a positive integer."
            )

        self.feature_function_ = None
        self.program_ = None
        self.is_fitted_ = False

    @staticmethod
    def _compile_program(
        program_text: str,
    ):
        """
        Compile EOH-generated Python source and return its
        extract_features function.
        """
        if not isinstance(
            program_text,
            str,
        ):
            raise TypeError(
                "EOH program must be supplied as Python source text."
            )

        program_text = (
            program_text.strip()
        )

        if not program_text:
            raise ValueError(
                "EOH program text is empty."
            )

        if "def extract_features" not in program_text:
            raise ValueError(
                "EOH program must define a function named "
                "extract_features."
            )

        namespace = {
            "np": np,
        }

        try:
            compiled_program = compile(
                program_text,
                "<eoh_feature_program>",
                "exec",
            )

            exec(
                compiled_program,
                namespace,
            )

        except Exception as error:
            raise ValueError(
                "The EOH program could not be compiled."
            ) from error

        feature_function = namespace.get(
            "extract_features"
        )

        if not callable(
            feature_function
        ):
            raise ValueError(
                "The EOH program does not define a callable "
                "extract_features function."
            )

        return feature_function

    def _validate_feature_vector(
        self,
        values,
        image_index: int | None = None,
    ) -> np.ndarray:
        """
        Validate one feature vector produced by the selected
        EOH program.
        """
        vector = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if vector.shape != (
            self.n_features,
        ):
            location = (
                ""
                if image_index is None
                else f" for image {image_index}"
            )

            raise ValueError(
                "EOH feature extractor produced shape "
                f"{vector.shape}{location}. "
                f"Expected ({self.n_features},)."
            )

        if not np.all(
            np.isfinite(vector)
        ):
            location = (
                ""
                if image_index is None
                else f" for image {image_index}"
            )

            raise ValueError(
                "EOH feature extractor produced NaN or "
                f"infinite values{location}."
            )

        return vector

    def set_program(
        self,
        program_text: str,
    ):
        """
        Supply or replace the selected EOH program.
        """
        self.program_text = (
            program_text.strip()
        )

        self.feature_function_ = None
        self.program_ = None
        self.is_fitted_ = False

        return self

    def load_program(
        self,
        path: str | Path,
    ):
        """
        Load a selected EOH program from a Python file.
        """
        program_path = Path(
            path
        )

        if not program_path.exists():
            raise FileNotFoundError(
                f"EOH program file not found: {program_path}"
            )

        if not program_path.is_file():
            raise ValueError(
                f"Expected a file, received: {program_path}"
            )

        program_text = (
            program_path.read_text(
                encoding="utf-8"
            )
        )

        return self.set_program(
            program_text
        )

    def save_program(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save the selected EOH program to a Python file.
        """
        if not self.program_text:
            raise RuntimeError(
                "No EOH program is available to save."
            )

        program_path = Path(
            path
        )

        program_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        program_path.write_text(
            self.program_text,
            encoding="utf-8",
        )

        return program_path

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Compile and validate the selected EOH program.

        Unlike GP, this method does not perform the EOH/LLM
        search itself. The search is performed separately
        using EOHFeatureExtractionProblem.

        The selected program is then supplied to this class.

        X and y are accepted to maintain the common MARS
        BaseFeatureExtractor interface.
        """
        X = np.asarray(
            X
        )

        y = np.asarray(
            y
        )

        if X.ndim not in {
            3,
            4,
        }:
            raise ValueError(
                "EOHExtractor expects image data with shape "
                "(n_samples, height, width) or "
                "(n_samples, height, width, channels). "
                f"Received {X.shape}."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number "
                "of samples."
            )

        if len(X) == 0:
            raise ValueError(
                "Cannot fit EOHExtractor on an empty dataset."
            )

        if not self.program_text:
            raise RuntimeError(
                "No selected EOH program has been supplied. "
                "Use program_text=..., set_program(...), "
                "or load_program(...) first."
            )

        feature_function = (
            self._compile_program(
                self.program_text
            )
        )

        # Contract test using one training image.
        sample_features = (
            feature_function(
                X[0]
            )
        )

        self._validate_feature_vector(
            sample_features,
            image_index=0,
        )

        self.feature_function_ = (
            feature_function
        )

        self.program_ = (
            self.program_text
        )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the selected EOH program to images.

        Returns
        -------
        feature_matrix:
            Array with shape:

                (n_samples, n_features)
        """
        if (
            not self.is_fitted_
            or self.feature_function_ is None
        ):
            raise RuntimeError(
                "EOHExtractor must be fitted before "
                "calling transform()."
            )

        X = np.asarray(
            X
        )

        if X.ndim not in {
            3,
            4,
        }:
            raise ValueError(
                "Expected image data with shape "
                "(n_samples, height, width) or "
                "(n_samples, height, width, channels). "
                f"Received {X.shape}."
            )

        rows = []

        for image_index, image in enumerate(
            X
        ):
            try:
                values = (
                    self.feature_function_(
                        image
                    )
                )

            except Exception as error:
                raise ValueError(
                    "EOH feature extraction failed for "
                    f"image {image_index}."
                ) from error

            vector = (
                self._validate_feature_vector(
                    values,
                    image_index=image_index,
                )
            )

            rows.append(
                vector
            )

        feature_matrix = np.vstack(
            rows
        )

        expected_shape = (
            len(X),
            self.n_features,
        )

        if feature_matrix.shape != expected_shape:
            raise ValueError(
                "EOH produced an unexpected feature "
                f"matrix shape {feature_matrix.shape}. "
                f"Expected {expected_shape}."
            )

        return self.validate_feature_matrix(
            feature_matrix,
            n_samples=len(X),
        )