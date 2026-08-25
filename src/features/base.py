from abc import ABC, abstractmethod

import numpy as np


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all MARS feature extractors.

    Every feature extraction method must implement the same
    fit/transform interface so that different extractors can
    be used by the same evaluation pipeline.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Learn the feature extraction method from training data.

        Parameters
        ----------
        X:
            Training images.

        y:
            Training labels.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Transform images into a feature matrix.

        Parameters
        ----------
        X:
            Images to transform.

        Returns
        -------
        features:
            Array with shape:

                (n_samples, n_features)
        """
        raise NotImplementedError

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the extractor and transform the training data.
        """
        self.fit(X, y)

        return self.transform(X)

    @staticmethod
    def validate_feature_matrix(
        features: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Validate the standard output produced by a feature
        extractor.
        """
        features = np.asarray(
            features,
            dtype=np.float64,
        )

        if features.ndim != 2:
            raise ValueError(
                "Feature extractor must return a 2D array. "
                f"Received shape {features.shape}."
            )

        if features.shape[0] != n_samples:
            raise ValueError(
                "Feature matrix has the wrong number of samples. "
                f"Expected {n_samples}, "
                f"received {features.shape[0]}."
            )

        if features.shape[1] == 0:
            raise ValueError(
                "Feature extractor returned zero features."
            )

        if not np.all(np.isfinite(features)):
            raise ValueError(
                "Feature matrix contains NaN or infinite values."
            )

        return features