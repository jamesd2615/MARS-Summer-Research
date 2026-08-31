from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color, io, transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MVTEC_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mvtec_anomaly_detection"
)

IMAGE_SIZE = (64, 64)

NORMAL_LABEL = 0
ANOMALY_LABEL = 1


def _load_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load one MVTec image and convert it to the standard
    grayscale 64 x 64 MARS representation.
    """
    image = io.imread(
        image_path
    )

    if image.ndim == 3:
        image = color.rgb2gray(
            image
        )

    image = transform.resize(
        image,
        IMAGE_SIZE,
        anti_aliasing=True,
        preserve_range=False,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    return image


def load_mvtec_dataset():
    """
    Load the pooled MVTec AD test images as a binary dataset.

    Labels
    ------
    0:
        normal / good

    1:
        anomalous / any defect type

    Returns
    -------
    X:
        Image array with shape
        (n_samples, 64, 64).

    y:
        Binary label array.

    metadata:
        Dictionary describing the dataset configuration.
    """
    if not MVTEC_DIR.exists():
        raise FileNotFoundError(
            f"MVTec dataset directory not found: {MVTEC_DIR}"
        )

    images = []
    labels = []

    category_names = []
    defect_names = []

    category_dirs = sorted(
        path
        for path in MVTEC_DIR.iterdir()
        if path.is_dir()
    )

    if not category_dirs:
        raise FileNotFoundError(
            f"No MVTec categories found in: {MVTEC_DIR}"
        )

    for category_dir in category_dirs:
        test_dir = (
            category_dir
            / "test"
        )

        if not test_dir.exists():
            continue

        defect_dirs = sorted(
            path
            for path in test_dir.iterdir()
            if path.is_dir()
        )

        for defect_dir in defect_dirs:
            defect_name = defect_dir.name

            label = (
                NORMAL_LABEL
                if defect_name == "good"
                else ANOMALY_LABEL
            )

            image_paths = sorted(
                defect_dir.glob("*.png")
            )

            for image_path in image_paths:
                images.append(
                    _load_image(
                        image_path
                    )
                )

                labels.append(
                    label
                )

                category_names.append(
                    category_dir.name
                )

                defect_names.append(
                    defect_name
                )

    if not images:
        raise RuntimeError(
            "No MVTec test images were loaded."
        )

    X = np.stack(
        images
    ).astype(
        np.float32,
        copy=False,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    metadata = {
        "dataset_name": "mvtec",
        "dataset_directory": str(
            MVTEC_DIR
        ),
        "image_size": IMAGE_SIZE,
        "n_samples": int(
            len(y)
        ),
        "n_classes": int(
            len(np.unique(y))
        ),
        "normal_label": NORMAL_LABEL,
        "anomaly_label": ANOMALY_LABEL,
        "normal_samples": int(
            np.sum(
                y == NORMAL_LABEL
            )
        ),
        "anomaly_samples": int(
            np.sum(
                y == ANOMALY_LABEL
            )
        ),
        "categories": sorted(
            set(
                category_names
            )
        ),
        "category_names": category_names,
        "defect_names": defect_names,
        "source_split": "official_test",
        "representation": "grayscale_64x64",
    }

    return X, y, metadata