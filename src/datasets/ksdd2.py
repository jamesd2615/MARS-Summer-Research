from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, io, transform

from src.config import RAW_DATA_DIR


KSDD2_DIR = RAW_DATA_DIR / "KolektorSDD2"

IMAGE_SIZE = (64, 64)

NORMAL_LABEL = 0
DEFECT_LABEL = 1


def validate_directory(path: Path) -> None:
    """
    Confirm that a required directory exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required directory not found: {path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {path}"
        )


def discover_image_mask_pairs(
    dataset_dir: Path = KSDD2_DIR,
) -> pd.DataFrame:
    """
    Discover KSDD2 image files and their corresponding masks.

    Returns a metadata DataFrame containing image paths,
    mask paths and identifying information.
    """
    validate_directory(dataset_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    records = []

    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in image_extensions:
            continue

        name_lower = path.stem.lower()

        # Masks are handled separately and should not be
        # treated as ordinary input images.
        if (
            "mask" in name_lower
            or name_lower.endswith("_label")
            or name_lower.endswith("_gt")
        ):
            continue

        possible_masks = [
            path.with_name(f"{path.stem}_mask{path.suffix}"),
            path.with_name(f"{path.stem}_label{path.suffix}"),
            path.with_name(f"{path.stem}_gt{path.suffix}"),
        ]

        mask_path = None

        for candidate in possible_masks:
            if candidate.exists():
                mask_path = candidate
                break

        records.append(
            {
                "filename": path.name,
                "filepath": path,
                "mask_path": mask_path,
                "parent_folder": path.parent.name,
            }
        )

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise ValueError(
            f"No KSDD2 images were discovered in {dataset_dir}."
        )

    return manifest


def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    """
    Validate the discovered KSDD2 manifest.
    """
    required_columns = {
        "filename",
        "filepath",
        "mask_path",
        "parent_folder",
    }

    missing_columns = required_columns.difference(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing_columns)}"
        )

    if len(manifest) == 0:
        raise ValueError(
            "KSDD2 manifest is empty."
        )

    missing_images = [
        path
        for path in manifest["filepath"]
        if not Path(path).exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} image files are missing."
        )


def preprocess_image(
    image: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Convert an image to grayscale, resize it and return
    a float32 array in the range [0, 1].
    """
    processed = np.asarray(image)

    if processed.ndim == 3:
        if processed.shape[-1] not in {3, 4}:
            raise ValueError(
                f"Unexpected image shape: {processed.shape}"
            )

        processed = color.rgb2gray(processed)

    elif processed.ndim != 2:
        raise ValueError(
            f"Expected grayscale or colour image, "
            f"received shape {processed.shape}."
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
            f"Expected image shape {image_size}, "
            f"received {processed.shape}."
        )

    if not np.all(np.isfinite(processed)):
        raise ValueError(
            "Preprocessed image contains non-finite values."
        )

    return processed


def derive_label(
    mask_path: Path | None,
) -> int:
    """
    Convert a KSDD2 segmentation mask into a binary
    normal/defect label.

    0 = normal
    1 = defect
    """
    if mask_path is None:
        return NORMAL_LABEL

    mask_path = Path(mask_path)

    if not mask_path.exists():
        return NORMAL_LABEL

    mask = io.imread(mask_path)
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[..., 0]

    if np.any(mask > 0):
        return DEFECT_LABEL

    return NORMAL_LABEL


def load_ksdd2_dataset(
    dataset_dir: Path = KSDD2_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and preprocess the complete KSDD2 dataset.

    Returns
    -------
    X_images:
        Image array with shape
        (n_samples, 64, 64).

    y:
        Binary labels where:
        0 = normal
        1 = defect.

    metadata:
        Metadata for every image.
    """
    manifest = discover_image_mask_pairs(
        dataset_dir
    )

    validate_manifest(
        manifest
    )

    images = []
    labels = []

    for row in manifest.itertuples(index=False):

        try:
            image = io.imread(
                row.filepath
            )

            processed = preprocess_image(
                image
            )

        except Exception as error:
            raise ValueError(
                f"Failed to load/preprocess "
                f"{row.filepath}"
            ) from error

        label = derive_label(
            row.mask_path
        )

        images.append(
            processed
        )

        labels.append(
            label
        )

    X_images = np.stack(
        images
    ).astype(np.float32)

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    validate_loaded_dataset(
        X_images,
        y,
    )

    metadata = manifest.copy()

    metadata["label"] = y

    return X_images, y, metadata


def validate_loaded_dataset(
    X_images: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate the arrays returned by the KSDD2 loader.
    """
    if X_images.ndim != 3:
        raise ValueError(
            f"Expected X_images to have 3 dimensions, "
            f"received shape {X_images.shape}."
        )

    if X_images.shape[1:] != IMAGE_SIZE:
        raise ValueError(
            f"Expected image size {IMAGE_SIZE}, "
            f"received {X_images.shape[1:]}."
        )

    if len(X_images) != len(y):
        raise ValueError(
            "Number of images and labels does not match."
        )

    if not np.all(np.isfinite(X_images)):
        raise ValueError(
            "Loaded image array contains non-finite values."
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset(
        {NORMAL_LABEL, DEFECT_LABEL}
    ):
        raise ValueError(
            f"Unexpected labels found: {unique_labels}"
        )