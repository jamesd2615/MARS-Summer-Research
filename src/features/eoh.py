from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from eoh import EoH, LLMConfig

from src.features.base import BaseFeatureExtractor
from src.features.eoh_problem import EOHFeatureExtractionProblem


class EOHExtractor(BaseFeatureExtractor):
    """
    Leakage-safe EOH feature extractor.

    In final nested-CV mode, fit(X, y) performs a fresh EOH search using
    only the current OUTER training fold. Candidate programs are evaluated
    on an internal stratified train/validation split. The selected program
    is then compiled and used by the common downstream Linear SVM.

    The outer validation fold is never supplied to the EOH search.

    A pre-selected program can still be supplied for legacy/smoke-test use,
    but final experiments should use search_on_fit=True.
    """

    def __init__(
        self,
        program_text: str | None = None,
        *,
        n_features: int = 8,
        search_on_fit: bool = False,
        dataset_name: str | None = None,
        seed: int = 42,
        population_size: int = 5,
        generations: int = 1,
        validation_size: float = 0.20,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        timeout: int = 180,
        max_seconds_per_evaluation: float = 150.0,
    ):
        self.program_text = program_text.strip() if program_text is not None else None
        self.n_features = int(n_features)
        if self.n_features <= 0:
            raise ValueError("n_features must be a positive integer.")

        self.search_on_fit = bool(search_on_fit)
        self.dataset_name = dataset_name.lower() if dataset_name else None
        self.seed = int(seed)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.validation_size = float(validation_size)
        self.num_samplers = int(num_samplers)
        self.num_evaluators = int(num_evaluators)
        self.timeout = int(timeout)
        self.max_seconds_per_evaluation = float(max_seconds_per_evaluation)

        # Filled in by the outer CV runner so paths are collision-free.
        self.outer_seed = int(seed)
        self.fold_number = None

        self.feature_function_ = None
        self.program_ = None
        self.is_fitted_ = False
        self.train_time_ = None
        self.train_fitness_ = None

    @staticmethod
    def _compile_program(program_text: str):
        if not isinstance(program_text, str):
            raise TypeError("EOH program must be supplied as Python source text.")
        program_text = program_text.strip()
        if not program_text:
            raise ValueError("EOH program text is empty.")
        if "def extract_features" not in program_text:
            raise ValueError("EOH program must define a function named extract_features.")

        namespace = {"np": np}
        try:
            compiled_program = compile(program_text, "<eoh_feature_program>", "exec")
            exec(compiled_program, namespace)
        except Exception as error:
            raise ValueError("The EOH program could not be compiled.") from error

        feature_function = namespace.get("extract_features")
        if not callable(feature_function):
            raise ValueError(
                "The EOH program does not define a callable extract_features function."
            )
        return feature_function

    def _validate_feature_vector(self, values, image_index: int | None = None) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (self.n_features,):
            location = "" if image_index is None else f" for image {image_index}"
            raise ValueError(
                f"EOH feature extractor produced shape {vector.shape}{location}. "
                f"Expected ({self.n_features},)."
            )
        if not np.all(np.isfinite(vector)):
            location = "" if image_index is None else f" for image {image_index}"
            raise ValueError(
                f"EOH feature extractor produced NaN or infinite values{location}."
            )
        return vector

    def set_program(self, program_text: str):
        self.program_text = program_text.strip()
        self.feature_function_ = None
        self.program_ = None
        self.is_fitted_ = False
        return self

    def load_program(self, path: str | Path):
        program_path = Path(path)
        if not program_path.exists():
            raise FileNotFoundError(f"EOH program file not found: {program_path}")
        if not program_path.is_file():
            raise ValueError(f"Expected a file, received: {program_path}")
        return self.set_program(program_path.read_text(encoding="utf-8"))

    def save_program(self, path: str | Path) -> Path:
        if not self.program_text:
            raise RuntimeError("No EOH program is available to save.")
        program_path = Path(path)
        program_path.parent.mkdir(parents=True, exist_ok=True)
        program_path.write_text(self.program_text.strip() + "\n", encoding="utf-8")
        return program_path

    def _build_llm_config(self) -> LLMConfig:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not available in the environment.")
        return LLMConfig(
            api_endpoint="api.deepseek.com",
            api_key=api_key,
            model="deepseek-chat",
            timeout=self.timeout,
        )

    def _search_paths(self) -> tuple[Path, Path]:
        if not self.dataset_name:
            raise RuntimeError("dataset_name is required for nested EOH search.")
        if self.fold_number is None:
            raise RuntimeError(
                "fold_number was not configured by the outer cross-validation runner."
            )

        base = (
    Path("results")
    / "eoh_search"
    / self.dataset_name
    / f"seed_{self.outer_seed}"
    / f"fold_{self.fold_number}"
     )
        program_path = (
            Path("programs")
            / "eoh"
            / self.dataset_name
            / f"seed_{self.outer_seed}"
            / f"fold_{self.fold_number}"
            / "selected_program.py"
        )
        return base, program_path

    def _run_nested_search(self, X: np.ndarray, y: np.ndarray) -> str:
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must lie strictly between 0 and 1.")
        if self.population_size < 1 or self.generations < 1:
            raise ValueError("EOH population_size and generations must be positive.")

        X_search_train, X_search_val, y_search_train, y_search_val = train_test_split(
            X,
            y,
            test_size=self.validation_size,
            random_state=self.seed,
            stratify=y,
        )

        output_dir, program_path = self._search_paths()

        # Prevent stale samples_best.json from a previous run being mistaken
        # for the result of this fold.
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        problem = EOHFeatureExtractionProblem(
            X_train=X_search_train,
            y_train=y_search_train,
            X_validation=X_search_val,
            y_validation=y_search_val,
            timeout=self.timeout,
            n_processes=1,
            max_seconds_per_evaluation=self.max_seconds_per_evaluation,
            diagnostic_subdir=(
                f"{self.dataset_name}_eoh_search/"
                f"seed_{self.outer_seed}/fold_{self.fold_number}"
            ),
        )

        search = EoH(
            llm=self._build_llm_config(),
            problem=problem,
            pop_size=self.population_size,
            n_pop=self.generations,
            num_samplers=self.num_samplers,
            num_evaluators=self.num_evaluators,
            output_dir=str(output_dir),
            debug=False,
        )

        print(
            "  Nested EOH search: "
            f"outer_seed={self.outer_seed}, fold={self.fold_number}, "
            f"search_seed={self.seed}, train={len(y_search_train)}, "
            f"validation={len(y_search_val)}, population={self.population_size}, "
            f"generations={self.generations}"
        )

        search.run()

        best_sample_path = output_dir / "results" / "samples" / "samples_best.json"
        if not best_sample_path.exists():
            raise FileNotFoundError(f"Best EOH sample not found: {best_sample_path}")

        best_sample = json.loads(best_sample_path.read_text(encoding="utf-8"))
        program_code = best_sample.get("code")
        if not program_code:
            raise ValueError("The best EOH sample does not contain generated code.")

        fitness = best_sample.get("objective")
        if fitness is None:
            fitness = best_sample.get("fitness")
        try:
            self.train_fitness_ = None if fitness is None else float(fitness)
        except (TypeError, ValueError):
            self.train_fitness_ = None

        self.program_text = program_code.strip()
        self.save_program(program_path)
        print(f"  Selected EOH program saved: {program_path}")

        return self.program_text

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim not in {3, 4}:
            raise ValueError(
                "EOHExtractor expects image data with shape "
                "(n_samples, height, width) or "
                "(n_samples, height, width, channels). "
                f"Received {X.shape}."
            )
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X) == 0:
            raise ValueError("Cannot fit EOHExtractor on an empty dataset.")

        started = time.perf_counter()

        if self.search_on_fit:
            self._run_nested_search(X, y)
        elif not self.program_text:
            raise RuntimeError(
                "No selected EOH program has been supplied. For final nested-CV "
                "experiments construct EOHExtractor(search_on_fit=True, ...)."
            )

        feature_function = self._compile_program(self.program_text)
        sample_features = feature_function(X[0])
        self._validate_feature_vector(sample_features, image_index=0)

        self.feature_function_ = feature_function
        self.program_ = self.program_text
        self.is_fitted_ = True
        self.train_time_ = float(time.perf_counter() - started)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.feature_function_ is None:
            raise RuntimeError("EOHExtractor must be fitted before calling transform().")

        X = np.asarray(X)
        if X.ndim not in {3, 4}:
            raise ValueError(
                "Expected image data with shape (n_samples, height, width) or "
                f"(n_samples, height, width, channels). Received {X.shape}."
            )

        rows = []
        for image_index, image in enumerate(X):
            try:
                values = self.feature_function_(image)
            except Exception as error:
                raise ValueError(
                    f"EOH feature extraction failed for image {image_index}."
                ) from error
            rows.append(
                self._validate_feature_vector(values, image_index=image_index)
            )

        feature_matrix = np.vstack(rows)
        expected_shape = (len(X), self.n_features)
        if feature_matrix.shape != expected_shape:
            raise ValueError(
                f"EOH produced an unexpected feature matrix shape "
                f"{feature_matrix.shape}. Expected {expected_shape}."
            )

        return self.validate_feature_matrix(feature_matrix, n_samples=len(X))
