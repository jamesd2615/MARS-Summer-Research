from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import calculate_classification_metrics


def set_neural_seed(seed: int) -> None:
    """
    Set random seeds used by NumPy, Python and PyTorch.

    This improves reproducibility across neural baseline runs.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_image_tensor(X: np.ndarray) -> torch.Tensor:
    """
    Convert NumPy image arrays into PyTorch CNN format.

    Supported input shapes
    ----------------------
    Grayscale:
        (n_samples, height, width)

    RGB:
        (n_samples, height, width, 3)

    Returns
    -------
    torch.Tensor
        Tensor with shape:
            (n_samples, channels, height, width)
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim == 3:
        # Grayscale:
        # (N, H, W) -> (N, 1, H, W)
        X = X[:, None, :, :]

    elif X.ndim == 4:
        if X.shape[-1] == 3:
            # RGB:
            # (N, H, W, C) -> (N, C, H, W)
            X = np.transpose(
                X,
                (0, 3, 1, 2),
            )

        elif X.shape[1] in {1, 3}:
            # Already channel-first.
            pass

        else:
            raise ValueError(
                "4D image arrays must use either "
                "(N, H, W, 3) or (N, C, H, W) format."
            )

    else:
        raise ValueError(
            "Image array must have shape "
            "(N, H, W), (N, H, W, 3), "
            "or (N, C, H, W)."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Image data contains NaN or infinite values."
        )

    return torch.from_numpy(
        np.ascontiguousarray(X)
    )


def predict_neural_model(
    model: nn.Module,
    X: torch.Tensor,
    *,
    batch_size: int = 64,
    device: torch.device,
) -> np.ndarray:
    """
    Generate class predictions from a trained neural model.
    """

    dataset = TensorDataset(
        X
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(
                device
            )

            logits = model(
                inputs
            )

            batch_predictions = torch.argmax(
                logits,
                dim=1,
            )

            predictions.append(
                batch_predictions
                .cpu()
                .numpy()
            )

    return np.concatenate(
        predictions
    )


def run_neural_experiment(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dataset_name: str = "unknown",
    method_name: str = "cnn",
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """
    Train and evaluate one neural baseline experiment.

    Pipeline
    --------
    1. Validate and convert image data.
    2. Train the neural network on the training split.
    3. Predict training labels.
    4. Predict validation/test labels.
    5. Calculate the shared MARS classification metrics.
    6. Return a standardised results dictionary.

    Notes
    -----
    This function deliberately does not perform cross-validation.
    Cross-validation is handled separately so that each fold receives
    a freshly initialised model.
    """

    if epochs < 1:
        raise ValueError(
            "epochs must be at least 1."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    X_train = np.asarray(
        X_train
    )

    X_test = np.asarray(
        X_test
    )

    y_train = np.asarray(
        y_train
    )

    y_test = np.asarray(
        y_test
    )

    if y_train.ndim != 1:
        raise ValueError(
            "Training labels must be 1D."
        )

    if y_test.ndim != 1:
        raise ValueError(
            "Test labels must be 1D."
        )

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

    set_neural_seed(
        seed
    )

    X_train_tensor = prepare_image_tensor(
        X_train
    )

    X_test_tensor = prepare_image_tensor(
        X_test
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    # CPU is currently the expected environment for the MARS project.
    # This remains device-agnostic in case CUDA is available later.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    training_start = time.perf_counter()

    training_loss_history: list[float] = []

    for _ in range(
        epochs
    ):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )

            optimiser.zero_grad()

            logits = model(
                inputs
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimiser.step()

            batch_samples = int(
                targets.shape[0]
            )

            total_loss += (
                float(loss.item())
                * batch_samples
            )

            total_samples += batch_samples

        mean_epoch_loss = total_loss / total_samples

        training_loss_history.append(
            float(mean_epoch_loss)
        )

    model_training_time = (
        time.perf_counter()
        - training_start
    )

    # -------------------------------------------------------
    # Predictions
    # -------------------------------------------------------

    train_predictions = predict_neural_model(
        model,
        X_train_tensor,
        batch_size=batch_size,
        device=device,
    )

    test_predictions = predict_neural_model(
        model,
        X_test_tensor,
        batch_size=batch_size,
        device=device,
    )

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------

    train_metrics = calculate_classification_metrics(
        y_train,
        train_predictions,
    )

    test_metrics = calculate_classification_metrics(
        y_test,
        test_predictions,
    )

    # -------------------------------------------------------
    # Standardised result
    # -------------------------------------------------------

    results = {
        "dataset": dataset_name,
        "method": method_name,
        "seed": int(seed),

        "n_train": int(
            len(y_train)
        ),

        "n_test": int(
            len(y_test)
        ),

        "epochs": int(
            epochs
        ),

        "batch_size": int(
            batch_size
        ),

        "learning_rate": float(
            learning_rate
        ),

        "model_training_time": float(
            model_training_time
        ),

        "device": str(
            device
        ),

        "training_loss_history": training_loss_history,
        "initial_training_loss": float(training_loss_history[0]),
        "final_training_loss": float(training_loss_history[-1]),
        "training_loss_change": float(
            training_loss_history[-1] - training_loss_history[0]
        ),

        "train_metrics": train_metrics,
        "test_metrics": test_metrics,

        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
    }

    return results