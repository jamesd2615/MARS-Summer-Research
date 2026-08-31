from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from eoh import EoH, LLMConfig

from src.datasets.fei import load_fei_dataset
from src.datasets.ksdd2 import load_ksdd2_dataset
from src.datasets.mvtec import load_mvtec_dataset
from src.datasets.stl10 import load_stl10_dataset
from src.features.eoh_problem import EOHFeatureExtractionProblem


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


def create_search_split(
    X,
    y,
    *,
    validation_size: float,
    seed: int,
):
    """
    Create the stratified train/validation split used during EOH search.

    This split is used to evaluate candidate feature-extraction programs
    during the evolutionary search. It is separate from the later
    cross-validation evaluation of the selected program.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=seed,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val

def build_llm_config() -> LLMConfig:
    """
    Build the LLM configuration for genuine EOH search.

    The DeepSeek API key is read from the local environment and is never
    hard-coded into the repository.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not available in the environment."
        )

    return LLMConfig(
        api_endpoint="api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=180,
    )

def build_eoh_problem(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    dataset_name: str,
) -> EOHFeatureExtractionProblem:
    """
    Build the EOH feature-extraction search problem for one dataset.
    """
    return EOHFeatureExtractionProblem(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_val,
        y_validation=y_val,
        timeout=180,
        n_processes=1,
        max_seconds_per_evaluation=150.0,
        diagnostic_subdir=f"{dataset_name.lower()}_eoh_search",
    )

def build_eoh_search(
    llm_config: LLMConfig,
    problem: EOHFeatureExtractionProblem,
    *,
    dataset_name: str,
    population_size: int,
    generations: int,
    num_samplers: int,
    num_evaluators: int,
) -> EoH:
    """
    Construct the genuine EOH evolutionary search.
    """
    output_dir = (
        Path("results")
        / f"{dataset_name.lower()}_eoh_search"
    )

    return EoH(
        llm=llm_config,
        problem=problem,
        pop_size=population_size,
        n_pop=generations,
        num_samplers=num_samplers,
        num_evaluators=num_evaluators,
        output_dir=str(output_dir),
        debug=False,
    )

def save_best_eoh_program(
    *,
    dataset_name: str,
) -> Path:
    """
    Read the best EOH sample and save its generated code as a Python file.
    """
    search_dir = (
        Path("results")
        / f"{dataset_name.lower()}_eoh_search"
    )

    best_sample_path = (
        search_dir
        / "results"
        / "samples"
        / "samples_best.json"
    )

    if not best_sample_path.exists():
        raise FileNotFoundError(
            f"Best EOH sample not found: {best_sample_path}"
        )

    best_sample = json.loads(
        best_sample_path.read_text(
            encoding="utf-8"
        )
    )

    program_code = best_sample.get(
        "code"
    )

    if not program_code:
        raise ValueError(
            "The best EOH sample does not contain generated code."
        )

    program_path = (
        Path("programs")
        / "eoh"
        / dataset_name.lower()
        / "selected_program.py"
    )

    program_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    program_path.write_text(
        program_code.strip() + "\n",
        encoding="utf-8",
    )

    return program_path

def run_eoh_search(args):
    """
    Run a genuine EOH feature-extraction search for one dataset.
    """
    print("=" * 60)
    print("MARS Genuine EOH Search")
    print("=" * 60)

    print(f"Dataset:       {args.dataset}")
    print(f"Seed:          {args.seed}")
    print(f"Validation:    {args.validation_size}")
    print(f"Population:    {args.population}")
    print(f"Generations:   {args.generations}")
    print(f"Samplers:      {args.num_samplers}")
    print(f"Evaluators:    {args.num_evaluators}")

    print("=" * 60)

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    print("Loading dataset...")

    X, y, metadata = load_dataset(
        args.dataset
    )

    print(f"Loaded {len(y)} samples.")

    # -------------------------------------------------------
    # Search train/validation split
    # -------------------------------------------------------

    X_train, X_val, y_train, y_val = create_search_split(
        X,
        y,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    print(
        f"Search training samples:   {len(y_train)}"
    )
    print(
        f"Search validation samples: {len(y_val)}"
    )

    # -------------------------------------------------------
    # EOH problem
    # -------------------------------------------------------

    print("Building EOH problem...")

    problem = build_eoh_problem(
        X_train,
        y_train,
        X_val,
        y_val,
        dataset_name=args.dataset,
    )

    # -------------------------------------------------------
    # LLM configuration
    # -------------------------------------------------------

    print("Loading LLM configuration...")

    llm_config = build_llm_config()

    # -------------------------------------------------------
    # EOH search
    # -------------------------------------------------------

    search = build_eoh_search(
        llm_config,
        problem,
        dataset_name=args.dataset,
        population_size=args.population,
        generations=args.generations,
        num_samplers=args.num_samplers,
        num_evaluators=args.num_evaluators,
    )

    print()
    print("Starting genuine EOH search...")
    print()

    search.run()

    # -------------------------------------------------------
    # Retrieve selected program
    # -------------------------------------------------------

    print()
    print("EOH search complete.")
    print("Retrieving best discovered program...")

    program_path = save_best_eoh_program(
        dataset_name=args.dataset,
    )

    print()
    print("=" * 60)
    print("EOH selected program saved")
    print("=" * 60)

    print(f"Program: {program_path}")

def build_parser():
    """
    Construct the command-line interface for genuine EOH search.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a genuine EOH feature-extraction search "
            "for a MARS dataset."
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
        help="Dataset used for EOH feature search.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the search train/validation split.",
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Fraction of data reserved for EOH search validation.",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=5,
        help="EOH population size.",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of EOH evolutionary generations.",
    )

    parser.add_argument(
        "--num-samplers",
        type=int,
        default=1,
        help="Number of concurrent EOH LLM samplers.",
    )

    parser.add_argument(
        "--num-evaluators",
        type=int,
        default=1,
        help="Number of concurrent EOH candidate evaluators.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_eoh_search(args)


if __name__ == "__main__":
    main()
