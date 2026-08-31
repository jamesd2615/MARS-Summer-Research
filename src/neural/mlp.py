from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """
    Small reusable multilayer perceptron for the MARS
    non-explainable reference baseline.

    The model supports both grayscale and RGB images by
    flattening the image into a 1D input vector.

    Parameters
    ----------
    input_shape:
        Shape of one image, excluding the batch dimension.

        Examples:
            (64, 64)
            (64, 64, 3)
            (1, 64, 64)
            (3, 64, 64)

    n_classes:
        Number of target classes.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        n_classes: int,
    ) -> None:
        super().__init__()

        if len(input_shape) < 2:
            raise ValueError(
                "input_shape must describe an image."
            )

        if n_classes < 2:
            raise ValueError(
                "n_classes must be at least 2."
            )

        self.input_shape = tuple(
            int(value)
            for value in input_shape
        )

        self.n_classes = int(
            n_classes
        )

        input_size = 1

        for dimension in self.input_shape:
            if dimension < 1:
                raise ValueError(
                    "All input dimensions must be positive."
                )

            input_size *= dimension

        self.input_size = int(
            input_size
        )

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                self.input_size,
                256,
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                self.n_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass through the MLP.

        Parameters
        ----------
        x:
            Batch of image tensors.

        Returns
        -------
        torch.Tensor
            Raw class logits with shape:
                (batch, n_classes)
        """

        if x.ndim < 3:
            raise ValueError(
                "MLP input must include a batch dimension "
                "and image dimensions."
            )

        return self.network(
            x
        )