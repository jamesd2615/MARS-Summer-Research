import random
import time

import numpy as np
from deap import base, creator, gp, tools

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

        if X.ndim != 3:
            raise ValueError(
                "OriginalGPExtractor expects image data with shape "
                "(n_samples, height, width). "
                f"Received {X.shape}."
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

        toolbox.register(
            "evaluate",
            eval_individual,
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

        if X.ndim != 3:
            raise ValueError(
                "Expected image data with shape "
                "(n_samples, height, width). "
                f"Received {X.shape}."
            )

        function = self.toolbox_.compile(
            expr=self.best_individual_
        )

        features = []

        for image in X:
            feature_vector = np.asarray(
                function(image),
                dtype=float,
            ).ravel()

            features.append(
                feature_vector
            )

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