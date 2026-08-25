from __future__ import annotations

import argparse

import numpy as np

from src.datasets.fei import load_fei_dataset
from src.datasets.ksdd2 import load_ksdd2_dataset
from src.evaluation.cross_validation import (
    run_stratified_cross_validation,
    summarise_cross_validation,
)
from src.evaluation.metadata import build_experiment_metadata
from src.evaluation.results import (
    save_cross_validation_results,
    save_experiment_metadata,
)
from src.features.gp_original import OriginalGPExtractor
from src.features.gp_modified import ModifiedGPExtractor
from src.features.eoh import EOHExtractor


def load_dataset(dataset_name: str):
    """
    Load a dataset using the standard MARS dataset interface.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "fei":
        return load_fei_dataset()

    if dataset_name == "ksdd2":
        return load_ksdd2_dataset()

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


def build_extractor(
    method_name: str,
    *,
    seed: int,
    population_size: int,
    generations: int,
    eoh_program_file: str | None = None,
):
    """
    Construct the requested feature extractor.
    """
    method_name = method_name.lower()

    if method_name == "gp_original":
        return OriginalGPExtractor(
            seed=seed,
            population_size=population_size,
            generations=generations,
            verbose=True,
        )

    if method_name == "gp_modified":
        return ModifiedGPExtractor(
            seed=seed,
            population_size=population_size,
            generations=generations,
            verbose=True,
        )

    if method_name == "eoh":
        if not eoh_program_file:
            raise ValueError(
                "EOH requires --eoh-program-file pointing to a saved "
                "feature-extraction Python file."
            )

        extractor = EOHExtractor(
            n_features=8,
        )

        extractor.load_program(
            eoh_program_file
        )

        return extractor

    raise ValueError(
        f"Unknown feature extraction method: {method_name}"
    )


def run_experiment(args):
    """
    Run one complete cross-validated MARS experiment.
    """

    print("=" * 60)
    print("MARS Modular Experiment")
    print("=" * 60)

    print(f"Dataset:     {args.dataset}")
    print(f"Method:      {args.method}")
    print(f"CV folds:    {args.folds}")
    print(f"Seed:        {args.seed}")
    if args.method == "eoh":
        print("Population:  N/A")
        print("Generations: N/A")
    else:
        print(f"Population:  {args.population}")
        print(f"Generations: {args.generations}")

    print("=" * 60)

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    print("Loading dataset...")

    X, y, metadata = load_dataset(
        args.dataset
    )

    print(
        f"Loaded {len(y)} samples."
    )

    # -------------------------------------------------------
    # Feature extractor
    # -------------------------------------------------------

    extractor = build_extractor(
        args.method,
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
        eoh_program_file=args.eoh_program_file,
    )

    # -------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------

    fold_results = run_stratified_cross_validation(
        extractor,
        X,
        y,
        dataset_name=args.dataset.upper(),
        method_name=args.method,
        n_splits=args.folds,
        random_state=args.seed,
    )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    summary = summarise_cross_validation(
        fold_results
    )

    print()
    print("=" * 60)
    print("Cross-validation summary")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    if args.method == "eoh":
        metadata_population = None
        metadata_generations = None
    else:
        metadata_population = args.population
        metadata_generations = args.generations

    experiment_metadata = build_experiment_metadata(
        dataset_name=args.dataset,
        method_name=args.method,
        seed=args.seed,
        folds=args.folds,
        population=metadata_population,
        generations=metadata_generations,
        n_samples=len(y),
        n_classes=len(np.unique(y)),
    )

    paths = save_cross_validation_results(
        fold_results,
        dataset_name=args.dataset,
        method_name=args.method,
    )

    metadata_path = save_experiment_metadata(
        experiment_metadata,
        dataset_name=args.dataset,
        method_name=args.method,
    )

    print()
    print("=" * 60)
    print("Results saved")
    print("=" * 60)

    print(
        f"Fold results: {paths['fold_results']}"
    )

    print(
        f"Summary:      {paths['summary']}"
    )

    print(
        f"Metadata:     {metadata_path}"
    )


def build_parser():
    """
    Construct the command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a modular MARS feature extraction experiment."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "fei",
            "ksdd2",
        ],
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "gp_original",
            "gp_modified",
            "eoh",
        ],
        help="Feature extraction method.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=500,
        help="GP population size.",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of GP generations.",
    )

    parser.add_argument(
        "--eoh-program-file",
        type=str,
        default=None,
        help=(
            "Path to a saved EOH feature-extraction Python file. "
            "Required when --method eoh is selected."
        ),
    )

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    run_experiment(
        args
    )


if __name__ == "__main__":
    main()