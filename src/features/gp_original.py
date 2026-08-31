import random
import time

import numpy as np
from deap import base, creator, gp, tools
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from src.features.base import BaseFeatureExtractor

from src.legacy.gp_original.gp_fr import gp_restrict
from src.legacy.gp_original.gp_fr.gp_fr_main import (
    build_pset,
    eval_individual,
    ea_simple,
    POPULATION,
    GENERATION,
    CX_PROB,
    MUT_PROB,
    ELITISM_PROB,
    INIT_MIN_DEPTH,
    INIT_MAX_DEPTH,
    TOURNAMENT_SIZE,
)


def _eval_individual_rgb(individual, toolbox, x_train, y_train):
    """Apply one evolved 2D GP program to R/G/B and concatenate features."""
    try:
        func = toolbox.compile(expr=individual)
        features = []
        for image in x_train:
            parts = [
                np.asarray(func(image[..., c]), dtype=float).ravel()
                for c in range(3)
            ]
            features.append(np.concatenate(parts))
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return (0.0,)
        features = preprocessing.MinMaxScaler().fit_transform(features)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr, va in skf.split(features, y_train):
            clf = LinearSVC(max_iter=5000)
            clf.fit(features[tr], y_train[tr])
            accs.append(clf.score(features[va], y_train[va]))
        return (round(100 * np.mean(accs), 2),)
    except Exception:
        return (0.0,)


class OriginalGPExtractor(BaseFeatureExtractor):
    """
    Modular wrapper around the original GP-FR implementation.

    The GP search is learned using only the training data.

    After fitting, the best evolved GP individual is used
    purely as a feature extractor. Classification is handled
    separately by the common MARS model/evaluation pipeline.
    """

    def __init__(
        self,
        seed: int = 0,
        population_size: int = POPULATION,
        generations: int = GENERATION,
        verbose: bool = True,
    ):
        self.seed = seed
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose

        self.toolbox_ = None
        self.best_individual_ = None
        self.program_ = None
        self.train_time_ = None
        self.train_fitness_ = None
        self.input_mode_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Evolve a GP feature extractor using training data only.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 3:
            self.input_mode_ = "grayscale"
        elif X.ndim == 4 and X.shape[-1] == 3:
            self.input_mode_ = "rgb"
        else:
            raise ValueError(
                "OriginalGPExtractor expects grayscale (n, h, w) or "
                f"RGB (n, h, w, 3) image data. Received {X.shape}."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if len(X) == 0:
            raise ValueError(
                "Cannot fit OriginalGPExtractor on an empty dataset."
            )

        random.seed(self.seed)
        np.random.seed(self.seed)

        img_height, img_width = X[0].shape[:2]

        pset = build_pset(
            img_height,
            img_width,
        )

        if not hasattr(
            creator,
            "FitnessMax_GPFR",
        ):
            creator.create(
                "FitnessMax_GPFR",
                base.Fitness,
                weights=(1.0,),
            )

        if not hasattr(
            creator,
            "Individual_GPFR",
        ):
            creator.create(
                "Individual_GPFR",
                gp.PrimitiveTree,
                fitness=creator.FitnessMax_GPFR,
            )

        toolbox = base.Toolbox()

        toolbox.register(
            "expr",
            gp_restrict.genHalfAndHalfMD,
            pset=pset,
            min_=INIT_MIN_DEPTH,
            max_=INIT_MAX_DEPTH,
        )

        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual_GPFR,
            toolbox.expr,
        )

        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
        )

        toolbox.register(
            "compile",
            gp.compile,
            pset=pset,
        )

        evaluator = _eval_individual_rgb if self.input_mode_ == "rgb" else eval_individual
        toolbox.register(
            "evaluate",
            evaluator,
            toolbox=toolbox,
            x_train=X,
            y_train=y,
        )

        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=TOURNAMENT_SIZE,
        )

        toolbox.register(
            "selectElitism",
            tools.selBest,
        )

        toolbox.register(
            "mate",
            gp.cxOnePoint,
        )

        toolbox.register(
            "expr_mut",
            gp_restrict.genFull,
            min_=0,
            max_=2,
        )

        toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=toolbox.expr_mut,
            pset=pset,
        )

        stats = tools.Statistics(
            key=lambda individual: individual.fitness.values
        )

        stats.register(
            "avg",
            np.mean,
        )

        stats.register(
            "max",
            np.max,
        )

        population = toolbox.population(
            n=self.population_size
        )

        hall_of_fame = tools.HallOfFame(5)

        start_time = time.time()

        population, _ = ea_simple(
            population,
            toolbox,
            CX_PROB,
            MUT_PROB,
            ELITISM_PROB,
            self.generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=self.verbose,
        )

        self.train_time_ = (
            time.time() - start_time
        )

        self.toolbox_ = toolbox

        self.best_individual_ = (
            hall_of_fame[0]
        )

        self.program_ = str(
            self.best_individual_
        )

        self.train_fitness_ = (
            self.best_individual_.fitness.values[0]
        )

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the evolved GP program to images and return
        the resulting feature matrix.
        """
        if (
            self.toolbox_ is None
            or self.best_individual_ is None
        ):
            raise RuntimeError(
                "OriginalGPExtractor must be fitted before "
                "calling transform()."
            )

        X = np.asarray(X)

        if self.input_mode_ == "grayscale" and X.ndim != 3:
            raise ValueError(f"Expected grayscale (n, h, w); received {X.shape}.")
        if self.input_mode_ == "rgb" and (X.ndim != 4 or X.shape[-1] != 3):
            raise ValueError(f"Expected RGB (n, h, w, 3); received {X.shape}.")

        function = self.toolbox_.compile(
            expr=self.best_individual_
        )

        features = []

        for image in X:
            if self.input_mode_ == "rgb":
                feature_vector = np.concatenate([
                    np.asarray(function(image[..., c]), dtype=float).ravel()
                    for c in range(3)
                ])
            else:
                feature_vector = np.asarray(function(image), dtype=float).ravel()

            features.append(feature_vector)

        try:
            feature_matrix = np.asarray(
                features,
                dtype=float,
            )

        except ValueError as error:
            raise ValueError(
                "The evolved GP program produced feature vectors "
                "with inconsistent dimensions."
            ) from error

        if feature_matrix.ndim == 1:
            feature_matrix = (
                feature_matrix.reshape(-1, 1)
            )

        return self.validate_feature_matrix(
            feature_matrix,
            n_samples=len(X),
        )