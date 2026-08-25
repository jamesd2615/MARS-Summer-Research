"""
Restricted Modified GP-FR-8 for image classification.

This module is derived from the Original GP-FR implementation based on:
    Fan et al., "Genetic Programming for Image Classification:
    A New Program Representation With Flexible Feature Reuse",
    IEEE TEVC, vol. 27, no. 3, 2023.

Restricted design
-----------------
Each valid GP individual must return exactly eight finite scalar features.

The root is the typed FC8 primitive. Its eight children each return one
scalar feature vector. Scalar feature primitives may operate directly on
the full image, on a detected region, or on a filtered image/region.

This modified restricted implementation additionally exposes two scalar
summaries of the rotation-invariant LBP descriptor:
    - rotation-invariant histogram entropy;
    - rotation-invariant peak probability.

The unrestricted Original GP implementation should remain unchanged in:
    baselines original/gp_fr/

This file belongs in:
    baselines restricted modified/gp_fr/gp_fr_main.py
"""

import argparse
import random
import time
import warnings

import numpy as np
from deap import base, creator, gp, tools
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from . import gp_fr_functions as F
from . import gp_restrict
from .gp_fr_types import Img, Int1, Int2, Int3, Region, Vector, Vector1

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Evolution parameters
# ---------------------------------------------------------------------------

POPULATION = 250
GENERATION = 10
CX_PROB = 0.80
MUT_PROB = 0.19
ELITISM_PROB = 0.01

INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
MAX_DEPTH = 10
TOURNAMENT_SIZE = 7

TARGET_FEATURE_DIMENSION = 8
CV_SPLITS = 5
CV_RANDOM_STATE = 42

# Unique names prevent collisions when unrestricted and restricted GP
# packages are loaded in the same Python kernel.
FITNESS_CREATOR_NAME = "FitnessMax_GPFR_RestrictedModified8"
INDIVIDUAL_CREATOR_NAME = "Individual_GPFR_RestrictedModified8"


CLASSIFIERS = {
    "SVM": lambda: LinearSVC(max_iter=5000),
    "LR": lambda: LogisticRegression(max_iter=1000, solver="lbfgs"),
    "RF": lambda: RandomForestClassifier(
        n_estimators=500,
        max_depth=100,
        n_jobs=-1,
        random_state=CV_RANDOM_STATE,
    ),
    "ERF": lambda: ExtraTreesClassifier(
        n_estimators=500,
        max_depth=100,
        n_jobs=-1,
        random_state=CV_RANDOM_STATE,
    ),
}


# ---------------------------------------------------------------------------
# Restricted primitive set
# ---------------------------------------------------------------------------

def build_pset(img_height, img_width):
    """
    Build the typed Restricted Modified GP-8 primitive set.

    Every valid complete tree has return type Vector1 and therefore must
    start with FC8. Each FC8 child has return type Vector and must be one
    scalar feature primitive.
    """
    required_modified_functions = (
        "rotation_invariant_entropy_feature",
        "rotation_invariant_peak_feature",
        "features_con8",
    )

    missing_modified_functions = [
        function_name
        for function_name in required_modified_functions
        if not hasattr(F, function_name)
    ]

    if missing_modified_functions:
        raise AttributeError(
            "Restricted Modified GP functions module is missing: "
            f"{missing_modified_functions}"
        )

    if img_height < 3 or img_width < 3:
        raise ValueError(
            "Restricted Modified GP requires images of at least "
            "3 x 3 pixels."
        )

    pset = gp.PrimitiveSetTyped(
        "GPFR_RESTRICTED_MODIFIED8",
        [Img],
        Vector1,
        prefix="ARG",
    )

    # Root: exactly eight scalar feature children.
    pset.addPrimitive(
        F.features_con8,
        [Vector] * TARGET_FEATURE_DIMENSION,
        Vector1,
        name="FC8",
    )

    # Scalar features operating on a detected/filtered region.
    region_scalar_primitives = [
        ("Mean_R", F.mean_feature),
        ("Std_R", F.std_feature),
        ("MinVal_R", F.min_feature),
        ("MaxVal_R", F.max_feature),
        ("Median_R", F.median_feature),
        ("Energy_R", F.energy_feature),
        ("GradX_R", F.gradient_x_feature),
        ("GradY_R", F.gradient_y_feature),
        ("LapVar_R", F.laplacian_variance_feature),
        ("Symmetry_R", F.symmetry_feature),
        (
            "RIFEntropy_R",
            F.rotation_invariant_entropy_feature,
        ),
        (
            "RIFPeak_R",
            F.rotation_invariant_peak_feature,
        ),
    ]

    for primitive_name, primitive_function in region_scalar_primitives:
        pset.addPrimitive(
            primitive_function,
            [Region],
            Vector,
            name=primitive_name,
        )

    # Scalar features operating on a full or filtered image.
    image_scalar_primitives = [
        ("Mean", F.mean_feature),
        ("Std", F.std_feature),
        ("MinVal", F.min_feature),
        ("MaxVal", F.max_feature),
        ("Median", F.median_feature),
        ("Energy", F.energy_feature),
        ("GradX", F.gradient_x_feature),
        ("GradY", F.gradient_y_feature),
        ("LapVar", F.laplacian_variance_feature),
        ("Symmetry", F.symmetry_feature),
        (
            "RIFEntropy",
            F.rotation_invariant_entropy_feature,
        ),
        (
            "RIFPeak",
            F.rotation_invariant_peak_feature,
        ),
    ]

    for primitive_name, primitive_function in image_scalar_primitives:
        pset.addPrimitive(
            primitive_function,
            [Img],
            Vector,
            name=primitive_name,
        )

    # Region filters.
    pset.addPrimitive(F.med_filter, [Region], Region, name="Med_R")
    pset.addPrimitive(F.mean_filter, [Region], Region, name="MeanFilter_R")
    pset.addPrimitive(F.min_filter, [Region], Region, name="MinFilter_R")
    pset.addPrimitive(F.max_filter, [Region], Region, name="MaxFilter_R")
    pset.addPrimitive(F.gau_filter, [Region, Int1], Region, name="Gau_R")
    pset.addPrimitive(
        F.gau_d_filter,
        [Region, Int1, Int2, Int2],
        Region,
        name="GauD_R",
    )
    pset.addPrimitive(F.lap_filter, [Region], Region, name="Lap_R")
    pset.addPrimitive(F.log1_filter, [Region], Region, name="LoG1_R")
    pset.addPrimitive(F.log2_filter, [Region], Region, name="LoG2_R")
    pset.addPrimitive(F.sobel_filter, [Region], Region, name="Sobel_R")
    pset.addPrimitive(F.sobel_x_filter, [Region], Region, name="SobelX_R")
    pset.addPrimitive(F.sobel_y_filter, [Region], Region, name="SobelY_R")

    # Full-image filters.
    pset.addPrimitive(F.med_filter, [Img], Img, name="Med")
    pset.addPrimitive(F.mean_filter, [Img], Img, name="MeanFilter")
    pset.addPrimitive(F.min_filter, [Img], Img, name="MinFilter")
    pset.addPrimitive(F.max_filter, [Img], Img, name="MaxFilter")
    pset.addPrimitive(F.gau_filter, [Img, Int1], Img, name="Gau")
    pset.addPrimitive(
        F.gau_d_filter,
        [Img, Int1, Int2, Int2],
        Img,
        name="GauD",
    )
    pset.addPrimitive(F.lap_filter, [Img], Img, name="Lap")
    pset.addPrimitive(F.log1_filter, [Img], Img, name="LoG1")
    pset.addPrimitive(F.log2_filter, [Img], Img, name="LoG2")
    pset.addPrimitive(F.sobel_filter, [Img], Img, name="Sobel")
    pset.addPrimitive(F.sobel_x_filter, [Img], Img, name="SobelX")
    pset.addPrimitive(F.sobel_y_filter, [Img], Img, name="SobelY")

    # Region detection.
    pset.addPrimitive(
        F.region_r,
        [Img, Int3, Int3, Int3, Int3],
        Region,
        name="RegionR",
    )
    pset.addPrimitive(
        F.region_s,
        [Img, Int3, Int3, Int3],
        Region,
        name="RegionS",
    )

    # Terminals.
    pset.renameArguments(ARG0="Image")
    pset.addEphemeralConstant(
        "Sigma",
        lambda: random.randint(1, 3),
        Int1,
    )
    pset.addEphemeralConstant(
        "Order",
        lambda: random.randint(0, 2),
        Int2,
    )
    pset.addEphemeralConstant(
        "Pos",
        lambda: random.randint(
            0,
            max(img_width, img_height) - 3,
        ),
        Int3,
    )

    return pset


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def _validate_images_and_labels(images, labels, split_name):
    images = np.asarray(images)
    labels = np.asarray(labels)

    if images.ndim < 3:
        raise ValueError(
            f"{split_name} images must have shape "
            "(n_images, height, width)."
        )

    if len(images) == 0:
        raise ValueError(f"{split_name} contains no images.")

    if len(images) != len(labels):
        raise ValueError(
            f"{split_name} image and label counts differ: "
            f"{len(images)} images and {len(labels)} labels."
        )

    if not np.all(np.isfinite(images)):
        raise ValueError(
            f"{split_name} images contain NaN or infinite values."
        )

    return images, labels


def _validate_feature_vector(feature_vector, image_index, split_name):
    feature_vector = np.asarray(
        feature_vector,
        dtype=float,
    ).reshape(-1)

    if feature_vector.shape != (TARGET_FEATURE_DIMENSION,):
        raise ValueError(
            f"{split_name} image {image_index} returned "
            f"{feature_vector.shape}; expected "
            f"({TARGET_FEATURE_DIMENSION},)."
        )

    if not np.all(np.isfinite(feature_vector)):
        raise ValueError(
            f"{split_name} image {image_index} returned "
            "NaN or infinite features."
        )

    return feature_vector


def build_feature_matrix(
    toolbox,
    individual,
    images,
    split_name,
):
    """
    Compile one GP individual and evaluate it on every image.

    A valid Restricted Modified GP-8 individual must return exactly eight finite
    values for every image.
    """
    function = toolbox.compile(expr=individual)

    feature_rows = []

    for image_index, image in enumerate(images):
        feature_rows.append(
            _validate_feature_vector(
                function(image),
                image_index=image_index,
                split_name=split_name,
            )
        )

    feature_matrix = np.vstack(feature_rows).astype(
        float,
        copy=False,
    )

    expected_shape = (
        len(images),
        TARGET_FEATURE_DIMENSION,
    )

    if feature_matrix.shape != expected_shape:
        raise RuntimeError(
            f"{split_name} feature matrix has shape "
            f"{feature_matrix.shape}; expected {expected_shape}."
        )

    return feature_matrix


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def diagnose_random_feature_dimensions(
    x_train,
    sample_size=250,
    seed=42,
):
    """
    Generate random Restricted Modified GP trees and inspect their output dimensions.

    Under the restricted grammar, every successful complete tree should
    return exactly TARGET_FEATURE_DIMENSION values.
    """
    x_train = np.asarray(x_train)

    if x_train.ndim < 3 or len(x_train) == 0:
        raise ValueError(
            "x_train must contain at least one image."
        )

    if sample_size < 1:
        raise ValueError(
            "sample_size must be at least one."
        )

    random.seed(seed)
    np.random.seed(seed)

    image_height, image_width = x_train[0].shape[:2]
    diagnostic_pset = build_pset(image_height, image_width)

    diagnostic_toolbox = base.Toolbox()
    diagnostic_toolbox.register(
        "expr",
        gp_restrict.genHalfAndHalfMD,
        pset=diagnostic_pset,
        min_=INIT_MIN_DEPTH,
        max_=INIT_MAX_DEPTH,
    )
    diagnostic_toolbox.register(
        "compile",
        gp.compile,
        pset=diagnostic_pset,
    )

    observed_dimensions = []
    target_dimension_programs = []
    failure_details = []

    for individual_index in range(sample_size):
        individual = None

        try:
            individual = gp.PrimitiveTree(
                diagnostic_toolbox.expr()
            )

            function = diagnostic_toolbox.compile(
                expr=individual
            )

            feature_vector = np.asarray(
                function(x_train[0]),
                dtype=float,
            ).reshape(-1)

            if feature_vector.size == 0:
                raise ValueError(
                    "Program returned an empty feature vector."
                )

            if not np.all(np.isfinite(feature_vector)):
                raise ValueError(
                    "Program returned non-finite features."
                )

            dimension = int(feature_vector.size)
            observed_dimensions.append(dimension)

            if dimension == TARGET_FEATURE_DIMENSION:
                target_dimension_programs.append(
                    str(individual)
                )

        except Exception as error:
            failure_details.append(
                {
                    "individual_index": int(individual_index),
                    "program": (
                        str(individual)
                        if individual is not None
                        else None
                    ),
                    "error": str(error),
                }
            )

    if observed_dimensions:
        unique_dimensions, counts = np.unique(
            observed_dimensions,
            return_counts=True,
        )
        dimension_counts = {
            int(dimension): int(count)
            for dimension, count in zip(
                unique_dimensions,
                counts,
            )
        }
    else:
        dimension_counts = {}

    result = {
        "target_feature_dimension": TARGET_FEATURE_DIMENSION,
        "sample_size": int(sample_size),
        "successful_programs": len(observed_dimensions),
        "failed_programs": len(failure_details),
        "minimum_dimension": (
            int(min(observed_dimensions))
            if observed_dimensions
            else None
        ),
        "maximum_dimension": (
            int(max(observed_dimensions))
            if observed_dimensions
            else None
        ),
        "target_dimension_program_count": len(
            target_dimension_programs
        ),
        "dimension_counts": dimension_counts,
        "target_dimension_programs": target_dimension_programs,
        "failure_details": failure_details,
    }

    print("=" * 70)
    print("Restricted Modified GP-8 dimension diagnostic")
    print("=" * 70)
    print(
        f"Target feature dimension : "
        f"{TARGET_FEATURE_DIMENSION}"
    )
    print(
        f"Random programs requested: "
        f"{sample_size}"
    )
    print(
        f"Successful programs      : "
        f"{result['successful_programs']}"
    )
    print(
        f"Failed programs          : "
        f"{result['failed_programs']}"
    )
    print(
        f"Minimum dimension        : "
        f"{result['minimum_dimension']}"
    )
    print(
        f"Maximum dimension        : "
        f"{result['maximum_dimension']}"
    )
    print(
        f"Exactly {TARGET_FEATURE_DIMENSION} features"
        f"     : "
        f"{result['target_dimension_program_count']}"
    )
    print("=" * 70)

    if dimension_counts:
        print("\nObserved feature dimensions:")
        for dimension, count in sorted(
            dimension_counts.items()
        ):
            print(
                f"{dimension:>6} features : "
                f"{count:>4} programs"
            )

    if target_dimension_programs:
        print("\nExample eight-feature program:")
        print(target_dimension_programs[0])

    return result


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def eval_individual(
    individual,
    toolbox,
    x_train,
    y_train,
):
    """
    Evaluate one Restricted Modified GP-8 individual using stratified CV accuracy.

    Any program that fails the strict eight-feature contract receives zero
    fitness.
    """
    try:
        features = build_feature_matrix(
            toolbox=toolbox,
            individual=individual,
            images=x_train,
            split_name="training",
        )

        # Reject constant feature columns. They provide no discrimination
        # and can hide degenerate trees.
        constant_columns = np.where(
            np.isclose(
                np.std(features, axis=0),
                0.0,
            )
        )[0]

        if len(constant_columns) > 0:
            return (0.0,)

        scaler = preprocessing.MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        skf = StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=CV_RANDOM_STATE,
        )

        accuracies = []

        for train_indices, validation_indices in skf.split(
            features_scaled,
            y_train,
        ):
            classifier = LinearSVC(max_iter=5000)
            classifier.fit(
                features_scaled[train_indices],
                y_train[train_indices],
            )
            accuracies.append(
                classifier.score(
                    features_scaled[validation_indices],
                    y_train[validation_indices],
                )
            )

        accuracy = round(
            100.0 * float(np.mean(accuracies)),
            2,
        )

    except Exception:
        accuracy = 0.0

    return (accuracy,)


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

def ea_simple(
    population,
    toolbox,
    cxpb,
    mutpb,
    elitpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=True,
):
    logbook = tools.Logbook()
    logbook.header = (
        ["gen", "nevals"]
        + (stats.fields if stats else [])
    )

    invalid_individuals = [
        individual
        for individual in population
        if not individual.fitness.valid
    ]

    fitnesses = list(
        map(
            toolbox.evaluate,
            invalid_individuals,
        )
    )

    for individual, fitness in zip(
        invalid_individuals,
        fitnesses,
    ):
        individual.fitness.values = fitness

    if halloffame is not None:
        halloffame.update(population)

    record = (
        stats.compile(population)
        if stats
        else {}
    )

    logbook.record(
        gen=0,
        nevals=len(invalid_individuals),
        **record,
    )

    if verbose:
        print(logbook.stream)

    for generation in range(1, ngen + 1):
        elitism_count = max(
            1,
            int(elitpb * len(population)),
        )

        elites = toolbox.selectElitism(
            population,
            k=elitism_count,
        )
        elites = [
            toolbox.clone(individual)
            for individual in elites
        ]

        offspring = toolbox.select(
            population,
            len(population) - elitism_count,
        )
        offspring = [
            toolbox.clone(individual)
            for individual in offspring
        ]

        index = 0

        while index < len(offspring) - 1:
            if random.random() < (
                cxpb / (cxpb + mutpb)
            ):
                (
                    offspring[index],
                    offspring[index + 1],
                ) = toolbox.mate(
                    offspring[index],
                    offspring[index + 1],
                )

                del offspring[index].fitness.values
                del offspring[index + 1].fitness.values
                index += 2

            else:
                offspring[index], = toolbox.mutate(
                    offspring[index]
                )
                del offspring[index].fitness.values
                index += 1

        if index < len(offspring):
            offspring[index], = toolbox.mutate(
                offspring[index]
            )
            del offspring[index].fitness.values

        offspring = elites + offspring

        invalid_individuals = [
            individual
            for individual in offspring
            if not individual.fitness.valid
        ]

        fitnesses = list(
            map(
                toolbox.evaluate,
                invalid_individuals,
            )
        )

        for individual, fitness in zip(
            invalid_individuals,
            fitnesses,
        ):
            individual.fitness.values = fitness

        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)

        record = (
            stats.compile(population)
            if stats
            else {}
        )

        logbook.record(
            gen=generation,
            nevals=len(invalid_individuals),
            **record,
        )

        if verbose:
            print(logbook.stream)

    return population, logbook


# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------

def eval_test(
    toolbox,
    individual,
    x_train,
    y_train,
    x_test,
    y_test,
):
    """
    Evaluate the winning individual on the supplied development-validation
    partition.

    In the benchmark notebook, x_test/y_test refer to the benchmark
    validation split, not the locked held-out test split.
    """
    train_features = build_feature_matrix(
        toolbox=toolbox,
        individual=individual,
        images=x_train,
        split_name="training",
    )

    test_features = build_feature_matrix(
        toolbox=toolbox,
        individual=individual,
        images=x_test,
        split_name="validation",
    )

    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(
        train_features
    )
    test_scaled = scaler.transform(
        test_features
    )

    best_accuracy = 0.0
    best_classifier_name = ""
    best_classifier = None

    for classifier_name, classifier_factory in (
        CLASSIFIERS.items()
    ):
        try:
            classifier = classifier_factory()
            classifier.fit(
                train_scaled,
                y_train,
            )
            accuracy = round(
                100.0
                * classifier.score(
                    test_scaled,
                    y_test,
                ),
                2,
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier_name = classifier_name
                best_classifier = classifier

        except Exception:
            continue

    if best_classifier is None:
        raise RuntimeError(
            "No classifier completed Restricted Modified GP-8 "
            "validation successfully."
        )

    return (
        best_accuracy,
        best_classifier_name,
        best_classifier,
        scaler,
        train_features,
        test_features,
    )


# ---------------------------------------------------------------------------
# Toolbox and DEAP creator construction
# ---------------------------------------------------------------------------

def _get_or_create_deap_types():
    if not hasattr(
        creator,
        FITNESS_CREATOR_NAME,
    ):
        creator.create(
            FITNESS_CREATOR_NAME,
            base.Fitness,
            weights=(1.0,),
        )

    if not hasattr(
        creator,
        INDIVIDUAL_CREATOR_NAME,
    ):
        creator.create(
            INDIVIDUAL_CREATOR_NAME,
            gp.PrimitiveTree,
            fitness=getattr(
                creator,
                FITNESS_CREATOR_NAME,
            ),
        )

    return getattr(
        creator,
        INDIVIDUAL_CREATOR_NAME,
    )


def build_toolbox(
    x_train,
    y_train,
):
    x_train, y_train = _validate_images_and_labels(
        x_train,
        y_train,
        "training",
    )

    image_height, image_width = x_train[0].shape[:2]
    pset = build_pset(
        image_height,
        image_width,
    )

    individual_class = _get_or_create_deap_types()

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
        individual_class,
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
        x_train=x_train,
        y_train=y_train,
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

    # Prevent tree bloat beyond the configured maximum height.
    toolbox.decorate(
        "mate",
        gp.staticLimit(
            key=lambda individual: individual.height,
            max_value=MAX_DEPTH,
        ),
    )

    toolbox.decorate(
        "mutate",
        gp.staticLimit(
            key=lambda individual: individual.height,
            max_value=MAX_DEPTH,
        ),
    )

    return toolbox, pset


# ---------------------------------------------------------------------------
# Main Restricted Modified GP-8 entry point
# ---------------------------------------------------------------------------

def run_gp_fr(
    x_train,
    y_train,
    x_test,
    y_test,
    seed=0,
    verbose=True,
    population_size=None,
    generations=None,
):
    """
    Run Restricted Modified GP-8.

    population_size and generations are optional smoke-test overrides.
    Omitting them uses the full configured experiment.
    """
    random.seed(seed)
    np.random.seed(seed)

    x_train, y_train = _validate_images_and_labels(
        x_train,
        y_train,
        "training",
    )
    x_test, y_test = _validate_images_and_labels(
        x_test,
        y_test,
        "validation",
    )

    actual_population_size = (
        POPULATION
        if population_size is None
        else int(population_size)
    )

    actual_generations = (
        GENERATION
        if generations is None
        else int(generations)
    )

    if actual_population_size < 2:
        raise ValueError(
            "population_size must be at least two."
        )

    if actual_generations < 0:
        raise ValueError(
            "generations cannot be negative."
        )

    toolbox, pset = build_toolbox(
        x_train,
        y_train,
    )

    statistics = tools.Statistics(
        key=lambda individual: individual.fitness.values
    )
    statistics.register("avg", np.mean)
    statistics.register("max", np.max)

    population = toolbox.population(
        n=actual_population_size
    )

    hall_of_fame = tools.HallOfFame(5)

    training_started = time.perf_counter()

    population, logbook = ea_simple(
        population,
        toolbox,
        CX_PROB,
        MUT_PROB,
        ELITISM_PROB,
        actual_generations,
        stats=statistics,
        halloffame=hall_of_fame,
        verbose=verbose,
    )

    training_time = (
        time.perf_counter()
        - training_started
    )

    if len(hall_of_fame) == 0:
        raise RuntimeError(
            "Restricted Modified GP-8 produced an empty hall of fame."
        )

    best_individual = hall_of_fame[0]

    if best_individual.fitness.values[0] <= 0.0:
        raise RuntimeError(
            "Restricted Modified GP-8 did not produce a valid "
            "positive-fitness individual."
        )

    (
        validation_accuracy,
        classifier_name,
        classifier,
        scaler,
        train_features,
        validation_features,
    ) = eval_test(
        toolbox,
        best_individual,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    if train_features.shape[1] != TARGET_FEATURE_DIMENSION:
        raise RuntimeError(
            "Winning Restricted Modified GP individual violated "
            "the eight-feature contract."
        )

    if verbose:
        print(
            f"\n=== Restricted Modified GP-8 "
            f"Results (seed={seed}) ==="
        )
        print(
            f"Training time: "
            f"{training_time:.1f}s"
        )
        print(
            f"Best CV fitness: "
            f"{best_individual.fitness.values[0]:.2f}%"
        )
        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.2f}% "
            f"(classifier: {classifier_name})"
        )
        print(
            f"Feature dimension: "
            f"{train_features.shape[1]}"
        )
        print(
            f"Program: {best_individual}"
        )

    return {
        "test_acc": validation_accuracy,
        "classifier": classifier_name,
        "classifier_object": classifier,
        "scaler": scaler,
        "train_time": float(training_time),
        "train_fitness": float(
            best_individual.fitness.values[0]
        ),
        "feature_dim": int(
            train_features.shape[1]
        ),
        "target_feature_dim": (
            TARGET_FEATURE_DIMENSION
        ),
        "program": str(best_individual),
        "train_features": train_features,
        "test_features": validation_features,
        "population_size": actual_population_size,
        "generations": actual_generations,
        "seed": int(seed),
        "logbook": logbook,
        "pset": pset,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Restricted Modified GP-FR-8"
    )

    parser.add_argument(
        "--train_data",
        required=True,
    )
    parser.add_argument(
        "--train_label",
        required=True,
    )
    parser.add_argument(
        "--test_data",
        required=True,
    )
    parser.add_argument(
        "--test_label",
        required=True,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=POPULATION,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=GENERATION,
    )

    arguments = parser.parse_args()

    x_train = np.load(arguments.train_data)
    y_train = np.load(arguments.train_label)
    x_test = np.load(arguments.test_data)
    y_test = np.load(arguments.test_label)

    print(
        f"Train: {x_train.shape}, "
        f"Validation: {x_test.shape}"
    )

    results = []

    for run_index in range(arguments.runs):
        run_seed = arguments.seed + run_index

        print(
            f"\n--- Run {run_index + 1}/"
            f"{arguments.runs} "
            f"(seed={run_seed}) ---"
        )

        result = run_gp_fr(
            x_train,
            y_train,
            x_test,
            y_test,
            seed=run_seed,
            verbose=True,
            population_size=arguments.population_size,
            generations=arguments.generations,
        )

        results.append(result)

    accuracies = [
        result["test_acc"]
        for result in results
    ]

    print(
        f"\n=== Summary over "
        f"{len(results)} runs ==="
    )
    print(
        f"Mean ± Std: "
        f"{np.mean(accuracies):.2f} ± "
        f"{np.std(accuracies):.2f}"
    )
    print(
        f"Maximum: "
        f"{np.max(accuracies):.2f}"
    )


if __name__ == "__main__":
    main()
