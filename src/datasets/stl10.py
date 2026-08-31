from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.transform import resize


# -------------------------------------------------------
# Project paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STL10_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stl10_binary"
)


# -------------------------------------------------------
# Settings
# -------------------------------------------------------

ORIGINAL_IMAGE_SIZE = (96, 96)
IMAGE_SIZE = (64, 64)

N_CHANNELS = 3


# -------------------------------------------------------
# Binary readers
# -------------------------------------------------------

def _load_images(path: Path) -> np.ndarray:
    """
    Load STL-10 binary image data.

    STL-10 images are stored as 96x96 RGB images in binary format.
    The returned array has shape:

        (n_samples, 64, 64, 3)

    with floating-point values in [0, 1].
    """
    if not path.exists():
        raise FileNotFoundError(
            f"STL-10 image file not found: {path}"
        )

    raw = np.fromfile(
        path,
        dtype=np.uint8,
    )

    pixels_per_image = (
        N_CHANNELS
        * ORIGINAL_IMAGE_SIZE[0]
        * ORIGINAL_IMAGE_SIZE[1]
    )

    if raw.size % pixels_per_image != 0:
        raise ValueError(
            f"Unexpected STL-10 binary size in {path}"
        )

    images = raw.reshape(
        -1,
        N_CHANNELS,
        ORIGINAL_IMAGE_SIZE[0],
        ORIGINAL_IMAGE_SIZE[1],
    )

    # STL-10's binary storage requires the spatial axes
    # to be transposed when reconstructing the images.
    images = images.transpose(
        0,
        3,
        2,
        1,
    )

    # Resize from 96x96 RGB to 64x64 RGB.
    resized = np.empty(
        (
            images.shape[0],
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            N_CHANNELS,
        ),
        dtype=np.float32,
    )

    for i, image in enumerate(images):
        resized[i] = resize(
            image,
            (
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                N_CHANNELS,
            ),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    resized /= 255.0

    return resized


def _load_labels(path: Path) -> np.ndarray:
    """
    Load STL-10 class labels.

    Original STL-10 labels are 1-10.
    They are converted to 0-9 for the MARS pipeline.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"STL-10 label file not found: {path}"
        )

    labels = np.fromfile(
        path,
        dtype=np.uint8,
    ).astype(np.int64)

    return labels - 1


def _load_class_names() -> list[str]:
    """
    Load the ten STL-10 class names.
    """
    path = STL10_DIR / "class_names.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"STL-10 class-name file not found: {path}"
        )

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# -------------------------------------------------------
# Public dataset loader
# -------------------------------------------------------

def load_stl10_dataset():
    """
    Load the labelled STL-10 dataset for the MARS benchmark.

    The official labelled training and testing images are pooled
    so that the common MARS stratified cross-validation framework
    can be applied consistently.

    The 100,000 unlabeled STL-10 images are deliberately excluded.

    Returns
    -------
    X : np.ndarray
        RGB images with shape (n_samples, 64, 64, 3).

    y : np.ndarray
        Integer class labels from 0 to 9.

    metadata : dict
        Dataset information and sample provenance.
    """
    train_X = _load_images(
        STL10_DIR / "train_X.bin"
    )
    train_y = _load_labels(
        STL10_DIR / "train_y.bin"
    )

    test_X = _load_images(
        STL10_DIR / "test_X.bin"
    )
    test_y = _load_labels(
        STL10_DIR / "test_y.bin"
    )

    if len(train_X) != len(train_y):
        raise ValueError(
            "STL-10 training image/label counts do not match."
        )

    if len(test_X) != len(test_y):
        raise ValueError(
            "STL-10 testing image/label counts do not match."
        )

    X = np.concatenate(
        [train_X, test_X],
        axis=0,
    )

    y = np.concatenate(
        [train_y, test_y],
        axis=0,
    )

    class_names = _load_class_names()

    source_splits = (
        ["official_train"] * len(train_y)
        + ["official_test"] * len(test_y)
    )

    metadata = {
        "dataset_name": "stl10",
        "dataset_directory": str(STL10_DIR),
        "original_image_size": ORIGINAL_IMAGE_SIZE,
        "image_size": IMAGE_SIZE,
        "n_channels": N_CHANNELS,
        "representation": "rgb_64x64",
        "n_samples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "class_names": class_names,
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "source_splits": source_splits,
        "unlabeled_data_used": False,
    }

    return X, y, metadata