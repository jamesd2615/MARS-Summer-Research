from __future__ import annotations

from datetime import datetime, timezone
from platform import python_version

import numpy as np
import sklearn


def build_experiment_metadata(
    *,
    dataset_name: str,
    method_name: str,
    seed: int,
    folds: int,
    population: int | None,
    generations: int | None,
    n_samples: int,
    n_classes: int,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> dict:
    """
    Build reproducibility metadata for one MARS experiment.

    Supports both:
        - explainable evolutionary feature-extraction methods
        - neural baseline methods

    The returned dictionary is designed to be saved alongside
    fold-level and summary results.
    """

    metadata = {
        "dataset": str(dataset_name),
        "method": str(method_name),

        "seed": int(seed),
        "folds": int(folds),

        "population": (
            None
            if population is None
            else int(population)
        ),

        "generations": (
            None
            if generations is None
            else int(generations)
        ),

        "epochs": (
            None
            if epochs is None
            else int(epochs)
        ),

        "batch_size": (
            None
            if batch_size is None
            else int(batch_size)
        ),

        "learning_rate": (
            None
            if learning_rate is None
            else float(learning_rate)
        ),

        "n_samples": int(
            n_samples
        ),

        "n_classes": int(
            n_classes
        ),

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "python_version": python_version(),

        "numpy_version": np.__version__,

        "scikit_learn_version": sklearn.__version__,
    }

    # Record PyTorch only when a neural method is being used.
    if method_name.lower() in {
        "cnn",
        "mlp",
    }:
        import torch

        metadata["pytorch_version"] = torch.__version__

    return metadata