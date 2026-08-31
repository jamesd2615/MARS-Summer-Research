from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.evaluation.cross_validation import summarise_cross_validation


def _safe_name(value: str) -> str:
    """
    Convert a dataset or method name into a filesystem-safe folder name.
    """
    value = str(value).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def get_results_directory(
    dataset_name: str,
    method_name: str,
    seed: int | None = None,
) -> Path:
    """
    Return the standard results directory for one dataset/method run.

    When a seed is provided, results are stored under a seed-specific
    subdirectory so repeated runs do not overwrite one another.
    """
    dataset_folder = _safe_name(
        dataset_name
    )

    method_folder = _safe_name(
        method_name
    )

    output_dir = (
        RESULTS_DIR
        / dataset_folder
        / method_folder
    )

    if seed is not None:
        output_dir = (
            output_dir
            / f"seed_{int(seed)}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def save_cross_validation_results(
    fold_results: pd.DataFrame,
    *,
    dataset_name: str,
    method_name: str,
    seed: int | None = None,
) -> dict[str, Path]:
    """
    Save fold-level and summary cross-validation results.

    Files created
    -------------
    fold_results.csv
        One row per cross-validation fold.

    summary.csv
        Mean and standard deviation for each numeric metric.

    Returns
    -------
    dict
        Paths to the files that were created.
    """
    if fold_results.empty:
        raise ValueError(
            "Cannot save empty cross-validation results."
        )

    output_dir = get_results_directory(
        dataset_name,
        method_name,
        seed=seed,
    )

    fold_path = (
        output_dir
        / "fold_results.csv"
    )

    summary_path = (
        output_dir
        / "summary.csv"
    )

    fold_results.to_csv(
        fold_path,
        index=False,
    )

    summary = summarise_cross_validation(
        fold_results
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return {
        "directory": output_dir,
        "fold_results": fold_path,
        "summary": summary_path,
    }


def save_program_text(
    program_text: str,
    *,
    dataset_name: str,
    method_name: str,
    filename: str = "selected_program.py",
) -> Path:
    """
    Save an evolved/generated feature program alongside its results.
    """
    if not program_text:
        raise ValueError(
            "program_text cannot be empty."
        )

    output_dir = get_results_directory(
        dataset_name,
        method_name,
    )

    program_path = (
        output_dir
        / filename
    )

    program_path.write_text(
        program_text,
        encoding="utf-8",
    )

    return program_path

def save_experiment_metadata(
    metadata: dict,
    *,
    dataset_name: str,
    method_name: str,
    seed: int | None = None,
    filename: str = "metadata.json",
) -> Path:
    """
    Save experiment metadata alongside the result CSV files.
    """
    if not isinstance(metadata, dict):
        raise TypeError(
            "metadata must be provided as a dictionary."
        )

    output_dir = get_results_directory(
        dataset_name,
        method_name,
        seed=seed,
    )

    metadata_path = (
        output_dir
        / filename
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return metadata_path
