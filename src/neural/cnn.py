from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    Small reusable convolutional neural network for the MARS
    non-explainable reference baseline.

    The network supports:
        - grayscale images: 1 input channel
        - RGB images: 3 input channels
        - binary classification
        - multiclass classification

    Images are expected in PyTorch format:

        (batch_size, channels, height, width)

    The model returns raw class logits. Softmax is deliberately
    not applied here because PyTorch CrossEntropyLoss expects
    unnormalised logits.

    Parameters
    ----------
    in_channels:
        Number of image channels.
        Use 1 for grayscale and 3 for RGB.

    n_classes:
        Number of target classes.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        if in_channels not in {1, 3}:
            raise ValueError(
                "in_channels must be 1 for grayscale "
                "or 3 for RGB images."
            )

        if n_classes < 2:
            raise ValueError(
                "n_classes must be at least 2."
            )

        self.in_channels = int(
            in_channels
        )

        self.n_classes = int(
            n_classes
        )

        # ---------------------------------------------------
        # Convolutional feature learner
        # ---------------------------------------------------

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),

            # Makes the classifier independent of the exact
            # spatial image dimensions.
            nn.AdaptiveAvgPool2d(
                output_size=(4, 4),
            ),
        )

        # ---------------------------------------------------
        # Classification head
        # ---------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 4 * 4,
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
        Run a forward pass through the CNN.

        Parameters
        ----------
        x:
            Image tensor with shape:
                (batch, channels, height, width)

        Returns
        -------
        torch.Tensor
            Raw class logits with shape:
                (batch, n_classes)
        """

        if x.ndim != 4:
            raise ValueError(
                "CNN input must have shape "
                "(batch, channels, height, width)."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {x.shape[1]}."
            )

        x = self.features(
            x
        )

        return self.classifier(
            x
        )