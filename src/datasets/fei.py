from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, io, transform

from src.config import FEI_DIR


IMAGE_SIZE = (64, 64)
SELECTED_SUBJECTS = tuple(range(1, 11))
EXPECTED_IMAGES_PER_SUBJECT = 14
EXPECTED_SELECTED_IMAGE_COUNT = (
    len(SELECTED_SUBJECTS) * EXPECTED_IMAGES_PER_SUBJECT
)


def parse_fei_filename(image_path: Path) -> tuple[int, int]:
    """
    Extract the FEI subject ID and image number from a filename.

    Example:
        17-06.jpg -> subject_id=17, image_number=6
    """
    parts = image_path.stem.split("-")

    if len(parts) != 2:
        raise ValueError(
            f"Unexpected FEI filename format: {image_path.name}"
        )

    try:
        subject_id = int(parts[0])
        image_number = int(parts[1])
    except ValueError as error:
        raise ValueError(
            f"Could not parse subject or image number from "
            f"{image_path.name}."
        ) from error

    return subject_id, image_number


def build_fei_metadata(
    fei_dir: Path = FEI_DIR,
) -> pd.DataFrame:
    """
    Inspect the FEI directory and create metadata for the
    selected benchmark subjects.
    """
    if not fei_dir.exists():
        raise FileNotFoundError(
            f"FEI dataset directory not found: {fei_dir}"
        )

    image_paths = sorted(
        list(fei_dir.glob("*.jpg"))
        + list(fei_dir.glob("*.jpeg"))
        + list(fei_dir.glob("*.png"))
    )

    records = []

    for image_path in image_paths:
        subject_id, image_number = parse_fei_filename(image_path)

        records.append(
            {
                "filename": image_path.name,
                "filepath": image_path,
                "subject_id": subject_id,
                "image_number": image_number,
                "label": subject_id - 1,
            }
        )

    fei_metadata = (
        pd.DataFrame(records)
        .sort_values(["subject_id", "image_number"])
        .reset_index(drop=True)
    )

    subject_counts = fei_metadata.groupby("subject_id").size()

    if len(fei_metadata) != 700:
        raise ValueError(
            f"Expected 700 FEI images, found {len(fei_metadata)}."
        )

    if fei_metadata["subject_id"].nunique() != 50:
        raise ValueError("Expected 50 FEI subjects.")

    if not (subject_counts == EXPECTED_IMAGES_PER_SUBJECT).all():
        raise ValueError(
            "Every FEI subject should have 14 images."
        )

    benchmark_metadata = (
        fei_metadata[
            fei_metadata["subject_id"].isin(SELECTED_SUBJECTS)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(benchmark_metadata) != EXPECTED_SELECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED_IMAGE_COUNT} selected images, "
            f"found {len(benchmark_metadata)}."
        )

    return benchmark_metadata


def preprocess_image_array(
    image: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Convert an image to grayscale, resize it, and return
    a float32 array in the range [0, 1].
    """
    processed = np.asarray(image)

    if processed.ndim == 3:
        if processed.shape[-1] not in {3, 4}:
            raise ValueError(
                f"Expected RGB/RGBA input, received {processed.shape}."
            )

        processed = color.rgb2gray(processed)

    elif processed.ndim != 2:
        raise ValueError(
            f"Expected grayscale or colour image, "
            f"received {processed.shape}."
        )

    processed = transform.resize(
        processed,
        image_size,
        anti_aliasing=True,
        preserve_range=False,
    )

    processed = np.asarray(
        processed,
        dtype=np.float32,
    )

    processed = np.clip(
        processed,
        0.0,
        1.0,
    )

    if processed.shape != image_size:
        raise ValueError(
            f"Expected {image_size}, received {processed.shape}."
        )

    if not np.all(np.isfinite(processed)):
        raise ValueError(
            "Preprocessed image contains non-finite values."
        )

    return processed


def load_and_preprocess_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load and preprocess one FEI image.
    """
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    try:
        return preprocess_image_array(
            io.imread(image_path)
        )

    except Exception as error:
        raise ValueError(
            f"Failed to preprocess:\n{image_path}"
        ) from error


def load_fei_dataset(
    fei_dir: Path = FEI_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the selected FEI benchmark dataset.

    Returns
    -------
    X_images:
        Image array with shape (140, 64, 64).

    y:
        Integer class labels with shape (140,).

    metadata:
        Metadata describing each selected image.
    """
    metadata = build_fei_metadata(fei_dir)

    loaded_images = [
        load_and_preprocess_image(path)
        for path in metadata["filepath"]
    ]

    X_images = np.stack(
        loaded_images
    ).astype(np.float32)

    y = metadata["label"].to_numpy(
        dtype=np.int64
    )

    expected_shape = (
        EXPECTED_SELECTED_IMAGE_COUNT,
        *IMAGE_SIZE,
    )

    if X_images.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, "
            f"received {X_images.shape}."
        )

    if y.shape != (
        EXPECTED_SELECTED_IMAGE_COUNT,
    ):
        raise ValueError(
            f"Unexpected label shape: {y.shape}"
        )

    return X_images, y, metadata