from __future__ import annotations

import argparse

import numpy as np

from src.datasets.fei import load_fei_dataset
from src.datasets.ksdd2 import load_ksdd2_dataset
from src.datasets.mvtec import load_mvtec_dataset
from src.datasets.stl10 import load_stl10_dataset

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

from src.neural.cnn import SimpleCNN
from src.neural.mlp import SimpleMLP
from src.neural.cross_validation import (
    run_neural_cross_validation,
    summarise_neural_cross_validation,
)


FEATURE_METHODS = {
    "gp_original",
    "gp_modified",
    "eoh",
}

NEURAL_METHODS = {
    "cnn",
    "mlp",
}


def load_dataset(dataset_name: str):
    """
    Load a dataset using the standard MARS dataset interface.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "fei":
        return load_fei_dataset()

    if dataset_name == "ksdd2":
        return load_ksdd2_dataset()

    if dataset_name == "mvtec":
        return load_mvtec_dataset()

    if dataset_name == "stl10":
        return load_stl10_dataset()

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
    dataset_name: str | None = None,
    eoh_population: int = 5,
    eoh_generations: int = 1,
    eoh_validation_size: float = 0.20,
):
    """
    Construct one of the explainable feature extractors.
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
        # Final protocol: perform a fresh EOH search inside every outer
        # training fold. A pre-selected program remains available only
        # for legacy/smoke-test use when explicitly supplied.
        if eoh_program_file:
            extractor = EOHExtractor(
                n_features=8,
                search_on_fit=False,
            )
            extractor.load_program(eoh_program_file)
            return extractor

        return EOHExtractor(
            n_features=8,
            search_on_fit=True,
            dataset_name=dataset_name,
            seed=seed,
            population_size=eoh_population,
            generations=eoh_generations,
            validation_size=eoh_validation_size,
        )

    raise ValueError(
        f"Unknown feature extraction method: {method_name}"
    )


def infer_image_channels(
    X: np.ndarray,
) -> int:
    """
    Infer whether a dataset contains grayscale or RGB images.

    Supported shapes
    ----------------
    Grayscale:
        (N, H, W)
        (N, 1, H, W)

    RGB:
        (N, H, W, 3)
        (N, 3, H, W)
    """
    X = np.asarray(X)

    if X.ndim == 3:
        return 1

    if X.ndim == 4:
        if X.shape[-1] == 3:
            return 3

        if X.shape[1] in {1, 3}:
            return int(
                X.shape[1]
            )

    raise ValueError(
        "Could not infer image channels from dataset shape "
        f"{X.shape}. Expected grayscale or RGB image data."
    )


def infer_neural_input_shape(
    X: np.ndarray,
) -> tuple[int, int, int]:
    """
    Return the channel-first image shape used by the neural pipeline.

    Returns
    -------
    tuple[int, int, int]
        (channels, height, width)
    """
    X = np.asarray(X)

    if X.ndim == 3:
        return (
            1,
            int(X.shape[1]),
            int(X.shape[2]),
        )

    if X.ndim == 4:
        if X.shape[-1] == 3:
            return (
                3,
                int(X.shape[1]),
                int(X.shape[2]),
            )

        if X.shape[1] in {1, 3}:
            return (
                int(X.shape[1]),
                int(X.shape[2]),
                int(X.shape[3]),
            )

    raise ValueError(
        "Could not infer neural input shape from dataset shape "
        f"{X.shape}. Expected grayscale or RGB image data."
    )


def validate_neural_labels(
    y: np.ndarray,
) -> int:
    """
    Validate labels for PyTorch CrossEntropyLoss and return
    the number of classes.

    Neural labels must be integer class indices:
        0, 1, ..., n_classes - 1
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(
            "Neural baseline labels must be a 1D array."
        )

    unique_labels = np.unique(
        y
    )

    if len(unique_labels) < 2:
        raise ValueError(
            "Neural baseline requires at least two classes."
        )

    expected_labels = np.arange(
        len(unique_labels)
    )

    if not np.array_equal(
        unique_labels,
        expected_labels,
    ):
        raise ValueError(
            "Neural baseline requires contiguous integer labels "
            "starting at 0. "
            f"Received labels: {unique_labels.tolist()}"
        )

    return int(
        len(unique_labels)
    )


def run_experiment(args):
    """
    Run one complete cross-validated MARS experiment.

    Explainable methods
    -------------------
    GP / Modified GP / EOH:
        images
        -> feature extractor
        -> common Linear SVM
        -> metrics

    Neural methods
    --------------
    CNN / MLP:
        images
        -> neural network
        -> predictions
        -> metrics
    """

    method_name = args.method.lower()

    print("=" * 60)
    print("MARS Modular Experiment")
    print("=" * 60)

    print(f"Dataset:     {args.dataset}")
    print(f"Method:      {args.method}")
    print(f"CV folds:    {args.folds}")
    print(f"Seed:        {args.seed}")

    if method_name in {
        "gp_original",
        "gp_modified",
    }:
        print(f"Population:  {args.population}")
        print(f"Generations: {args.generations}")

    elif method_name == "eoh":
        if args.eoh_program_file:
            print("EOH mode:     pre-selected program (legacy/smoke test)")
            print(f"EOH program:  {args.eoh_program_file}")
        else:
            print("EOH mode:     nested leakage-safe search")
            print(f"Population:   {args.eoh_population}")
            print(f"Generations:  {args.eoh_generations}")
            print(f"Search val:   {args.eoh_validation_size}")

    elif method_name in NEURAL_METHODS:
        print(f"Epochs:      {args.epochs}")
        print(f"Batch size:  {args.batch_size}")
        print(
            f"Learning rate: {args.learning_rate}"
        )

    print("=" * 60)

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    print("Loading dataset...")

    X, y, _dataset_metadata = load_dataset(
        args.dataset
    )

    X = np.asarray(X)
    y = np.asarray(y)

    print(
        f"Loaded {len(y)} samples."
    )

    print(
        f"Image array shape: {X.shape}"
    )

    print(
        f"Classes: {np.unique(y).tolist()}"
    )

    # -------------------------------------------------------
    # Method-specific cross-validation
    # -------------------------------------------------------

    if method_name in FEATURE_METHODS:

        extractor = build_extractor(
            method_name,
            seed=args.seed,
            population_size=args.population,
            generations=args.generations,
            eoh_program_file=args.eoh_program_file,
            dataset_name=args.dataset,
            eoh_population=args.eoh_population,
            eoh_generations=args.eoh_generations,
            eoh_validation_size=args.eoh_validation_size,
        )

        fold_results = run_stratified_cross_validation(
            extractor,
            X,
            y,
            dataset_name=args.dataset.upper(),
            method_name=method_name,
            n_splits=args.folds,
            random_state=args.seed,
        )

        summary = summarise_cross_validation(
            fold_results
        )

    elif method_name in NEURAL_METHODS:

        n_classes = validate_neural_labels(
            y
        )

        if method_name == "cnn":
            in_channels = infer_image_channels(
                X
            )

            print(
                f"CNN input channels: {in_channels}"
            )

            print(
                f"CNN output classes: {n_classes}"
            )

            def model_builder():
                return SimpleCNN(
                    in_channels=in_channels,
                    n_classes=n_classes,
                )

        elif method_name == "mlp":
            input_shape = infer_neural_input_shape(
                X
            )

            print(
                f"MLP input shape: {input_shape}"
            )

            print(
                f"MLP output classes: {n_classes}"
            )

            def model_builder():
                return SimpleMLP(
                    input_shape=input_shape,
                    n_classes=n_classes,
                )

        fold_results = run_neural_cross_validation(
            model_builder,
            X,
            y,
            dataset_name=args.dataset.upper(),
            method_name=method_name,
            n_splits=args.folds,
            random_state=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        summary = summarise_neural_cross_validation(
            fold_results
        )

    else:
        raise ValueError(
            f"Unknown method: {method_name}"
        )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

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
    # Metadata
    # -------------------------------------------------------

    if method_name in {
        "gp_original",
        "gp_modified",
    }:
        metadata_population = args.population
        metadata_generations = args.generations

        metadata_epochs = None
        metadata_batch_size = None
        metadata_learning_rate = None

    elif method_name == "eoh":
        metadata_population = None
        metadata_generations = None

        metadata_epochs = None
        metadata_batch_size = None
        metadata_learning_rate = None

    else:
        metadata_population = None
        metadata_generations = None

        metadata_epochs = args.epochs
        metadata_batch_size = args.batch_size
        metadata_learning_rate = args.learning_rate

    experiment_metadata = build_experiment_metadata(
        dataset_name=args.dataset,
        method_name=method_name,
        seed=args.seed,
        folds=args.folds,
        population=metadata_population,
        generations=metadata_generations,
        epochs=metadata_epochs,
        batch_size=metadata_batch_size,
        learning_rate=metadata_learning_rate,
        n_samples=len(y),
        n_classes=len(np.unique(y)),
    )

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    paths = save_cross_validation_results(
        fold_results,
        dataset_name=args.dataset,
        method_name=method_name,
        seed=args.seed,
    )

    metadata_path = save_experiment_metadata(
        experiment_metadata,
        dataset_name=args.dataset,
        method_name=method_name,
        seed=args.seed,
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
            "Run a modular MARS explainable-feature or neural "
            "baseline experiment."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "fei",
            "ksdd2",
            "mvtec",
            "stl10",
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
            "cnn",
            "mlp",
        ],
        help="Method to evaluate.",
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

    # -------------------------------------------------------
    # GP options
    # -------------------------------------------------------

    parser.add_argument(
        "--population",
        type=int,
        default=500,
        help=(
            "GP population size. Used only by GP methods."
        ),
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help=(
            "Number of GP generations. Used only by GP methods."
        ),
    )

    # -------------------------------------------------------
    # EOH options
    # -------------------------------------------------------

    parser.add_argument(
        "--eoh-program-file",
        type=str,
        default=None,
        help=(
            "Optional pre-selected EOH program for legacy/smoke tests. "
            "For final EOH experiments omit this option so search occurs "
            "inside each outer training fold."
        ),
    )

    parser.add_argument(
        "--eoh-population",
        type=int,
        default=5,
        help="EOH population size used inside each outer training fold.",
    )

    parser.add_argument(
        "--eoh-generations",
        type=int,
        default=1,
        help="EOH generations used inside each outer training fold.",
    )

    parser.add_argument(
        "--eoh-validation-size",
        type=float,
        default=0.20,
        help="Fraction of each outer training fold used for EOH search validation.",
    )

    # -------------------------------------------------------
    # Neural options
    # -------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Number of neural-network training epochs per fold. "
            "Used only by neural methods."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Neural-network batch size. "
            "Used only by neural methods."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help=(
            "Adam learning rate for neural methods."
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
