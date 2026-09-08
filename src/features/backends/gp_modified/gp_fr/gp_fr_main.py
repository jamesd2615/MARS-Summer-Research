"""
GP-FR: Genetic Programming with Flexible Feature Reuse for Image Classification.

Reimplementation based on:
  Fan et al., "Genetic Programming for Image Classification: A New Program
  Representation With Flexible Feature Reuse", IEEE TEVC, vol. 27, no. 3, 2023.

Usage:
    python -m baselines.gp_fr.gp_fr_main \
        --train_data data/train_images.npy \
        --train_label data/train_labels.npy \
        --test_data data/test_images.npy \
        --test_label data/test_labels.npy \
        --runs 30 --seed 0
"""
import argparse
import random
import time
import warnings

import numpy as np
from deap import base, creator, tools, gp
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from . import gp_restrict
from .gp_fr_types import (
    Int1, Int2, Int3, Float1, Float2, Float3,
    Img, Region, Vector, Vector1,
)
from . import gp_fr_functions as F

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paper parameters (Section IV-C)
# ---------------------------------------------------------------------------
POPULATION = 250
GENERATION = 10
CX_PROB = 0.8
MUT_PROB = 0.19
ELITISM_PROB = 0.01
INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
MAX_DEPTH = 10
TOURNAMENT_SIZE = 7

CLASSIFIERS = {
    1: lambda: LinearSVC(max_iter=5000),
    2: lambda: LogisticRegression(max_iter=1000, solver='lbfgs'),
    3: lambda: RandomForestClassifier(n_estimators=500, max_depth=100, n_jobs=-1),
    4: lambda: ExtraTreesClassifier(n_estimators=500, max_depth=100, n_jobs=-1),
}


# ---------------------------------------------------------------------------
# Build the primitive set (Section III)
# ---------------------------------------------------------------------------
def build_pset(img_height, img_width):
    pset = gp.PrimitiveSetTyped('GPFR', [Img], Vector1, prefix='ARG')

    # --- Feature concatenation layer ---
    pset.addPrimitive(F.features_con, [Vector, Vector], Vector1, name='FC2')
    pset.addPrimitive(F.features_con, [Vector, Vector, Vector], Vector1, name='FC3')
    pset.addPrimitive(F.features_con, [Vector, Vector, Vector, Vector], Vector1, name='FC4')

    # --- Feature extraction on Region ---
    pset.addPrimitive(F.ulbp_feature, [Region], Vector, name='uLBP_R')
    pset.addPrimitive(F.sift_feature, [Region], Vector, name='SIFT_R')
    pset.addPrimitive(F.hog_feature, [Region], Vector, name='HOG_R')
    pset.addPrimitive(F.hist_feature, [Region], Vector, name='Hist_R')
    pset.addPrimitive(F.dif_feature, [Region], Vector, name='DIF_R')

    # Modified GP feature
    pset.addPrimitive(
        F.rotation_invariant_feature,
        [Region],
        Vector,
        name='RIF_R',
    )

    # --- Feature extraction on full/filtered image ---
    pset.addPrimitive(F.ulbp_feature, [Img], Vector, name='uLBP')
    pset.addPrimitive(F.sift_feature, [Img], Vector, name='SIFT')
    pset.addPrimitive(F.hog_feature, [Img], Vector, name='HOG')
    pset.addPrimitive(F.hist_feature, [Img], Vector, name='Hist')
    pset.addPrimitive(F.dif_feature, [Img], Vector, name='DIF')

    # Modified GP feature
    pset.addPrimitive(
        F.rotation_invariant_feature,
        [Img],
        Vector,
        name='RIF',
    )

    # --- Image filtering on Region (with feature reuse output) ---
    pset.addPrimitive(F.med_filter, [Region], Region, name='Med_R')
    pset.addPrimitive(F.mean_filter, [Region], Region, name='Mean_R')
    pset.addPrimitive(F.min_filter, [Region], Region, name='Min_R')
    pset.addPrimitive(F.max_filter, [Region], Region, name='Max_R')
    pset.addPrimitive(F.gau_filter, [Region, Int1], Region, name='Gau_R')
    pset.addPrimitive(F.gau_d_filter, [Region, Int1, Int2, Int2], Region, name='GauD_R')
    pset.addPrimitive(F.lap_filter, [Region], Region, name='Lap_R')
    pset.addPrimitive(F.log1_filter, [Region], Region, name='LoG1_R')
    pset.addPrimitive(F.log2_filter, [Region], Region, name='LoG2_R')
    pset.addPrimitive(F.sobel_filter, [Region], Region, name='Sobel_R')
    pset.addPrimitive(F.sobel_x_filter, [Region], Region, name='SobelX_R')
    pset.addPrimitive(F.sobel_y_filter, [Region], Region, name='SobelY_R')

    # --- Image filtering on full image ---
    pset.addPrimitive(F.med_filter, [Img], Img, name='Med')
    pset.addPrimitive(F.mean_filter, [Img], Img, name='Mean')
    pset.addPrimitive(F.min_filter, [Img], Img, name='Min')
    pset.addPrimitive(F.max_filter, [Img], Img, name='Max')
    pset.addPrimitive(F.gau_filter, [Img, Int1], Img, name='Gau')
    pset.addPrimitive(F.gau_d_filter, [Img, Int1, Int2, Int2], Img, name='GauD')
    pset.addPrimitive(F.lap_filter, [Img], Img, name='Lap')
    pset.addPrimitive(F.log1_filter, [Img], Img, name='LoG1')
    pset.addPrimitive(F.log2_filter, [Img], Img, name='LoG2')
    pset.addPrimitive(F.sobel_filter, [Img], Img, name='Sobel')
    pset.addPrimitive(F.sobel_x_filter, [Img], Img, name='SobelX')
    pset.addPrimitive(F.sobel_y_filter, [Img], Img, name='SobelY')

    # --- Region detection layer ---
    pset.addPrimitive(F.region_r, [Img, Int3, Int3, Int3, Int3], Region, name='RegionR')
    pset.addPrimitive(F.region_s, [Img, Int3, Int3, Int3], Region, name='RegionS')

    # --- Terminals ---
    pset.renameArguments(ARG0='Image')
    pset.addEphemeralConstant('Sigma', lambda: random.randint(1, 3), Int1)
    pset.addEphemeralConstant('Order', lambda: random.randint(0, 2), Int2)
    pset.addEphemeralConstant(
        'Pos',
        lambda: random.randint(0, max(img_width, img_height) - 3),
        Int3,
    )

    return pset
# ---------------------------------------------------------------------------
# Fitness evaluation (Section III-E): stratified 5-fold CV accuracy
# ---------------------------------------------------------------------------
def eval_individual(individual, toolbox, x_train, y_train):
    try:
        func = toolbox.compile(expr=individual)
        features = []
        for i in range(len(y_train)):
            feat = np.asarray(func(x_train[i]), dtype=float).ravel()
            features.append(feat)
        features = np.array(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return (0.0,)

        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for train_idx, val_idx in skf.split(features, y_train):
            clf = LinearSVC(max_iter=5000)
            clf.fit(features[train_idx], y_train[train_idx])
            accs.append(clf.score(features[val_idx], y_train[val_idx]))
        accuracy = round(100 * np.mean(accs), 2)
    except Exception:
        accuracy = 0.0
    return (accuracy,)


# ---------------------------------------------------------------------------
# Evolution loop (Algorithm 1)
# ---------------------------------------------------------------------------
def ea_simple(population, toolbox, cxpb, mutpb, elitpb, ngen,
              stats=None, halloffame=None, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        elitism_n = max(1, int(elitpb * len(population)))
        elites = toolbox.selectElitism(population, k=elitism_n)
        elites = [toolbox.clone(ind) for ind in elites]

        offspring = toolbox.select(population, len(population) - elitism_n)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # crossover and mutation
        i = 0
        while i < len(offspring) - 1:
            if random.random() < cxpb / (cxpb + mutpb):
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values, offspring[i + 1].fitness.values
                i += 2
            else:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
                i += 1
        if i < len(offspring):
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

        offspring = elites + offspring

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# ---------------------------------------------------------------------------
# Test evaluation with automatic classifier selection
# ---------------------------------------------------------------------------
def eval_test(toolbox, individual, x_train, y_train, x_test, y_test):
    func = toolbox.compile(expr=individual)

    train_feat = np.array([np.asarray(func(img), dtype=float).ravel() for img in x_train])
    test_feat = np.array([np.asarray(func(img), dtype=float).ravel() for img in x_test])

    scaler = preprocessing.MinMaxScaler()
    train_norm = scaler.fit_transform(train_feat)
    test_norm = scaler.transform(test_feat)

    best_acc = 0.0
    best_name = ''
    for name, clf_fn in [('SVM', lambda: LinearSVC(max_iter=5000)),
                          ('LR', lambda: LogisticRegression(max_iter=1000)),
                          ('RF', lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1)),
                          ('ERF', lambda: ExtraTreesClassifier(n_estimators=500, n_jobs=-1))]:
        try:
            clf = clf_fn()
            clf.fit(train_norm, y_train)
            acc = round(100 * clf.score(test_norm, y_test), 2)
            if acc > best_acc:
                best_acc = acc
                best_name = name
        except Exception:
            continue

    return best_acc, best_name, train_feat, test_feat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_gp_fr(x_train, y_train, x_test, y_test, seed=0, verbose=True):
    random.seed(seed)
    np.random.seed(seed)

    img_h, img_w = x_train[0].shape[:2]
    pset = build_pset(img_h, img_w)

    if not hasattr(creator, 'FitnessMax_GPFR'):
        creator.create('FitnessMax_GPFR', base.Fitness, weights=(1.0,))
        creator.create('Individual_GPFR', gp.PrimitiveTree, fitness=creator.FitnessMax_GPFR)

    toolbox = base.Toolbox()
    toolbox.register('expr', gp_restrict.genHalfAndHalfMD,
                     pset=pset, min_=INIT_MIN_DEPTH, max_=INIT_MAX_DEPTH)
    toolbox.register('individual', tools.initIterate, creator.Individual_GPFR, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)
    toolbox.register('evaluate', eval_individual,
                     toolbox=toolbox, x_train=x_train, y_train=y_train)
    toolbox.register('select', tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register('selectElitism', tools.selBest)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.register('expr_mut', gp_restrict.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register('avg', np.mean)
    stats_fit.register('max', np.max)

    pop = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(5)

    t0 = time.time()
    pop, log = ea_simple(pop, toolbox, CX_PROB, MUT_PROB, ELITISM_PROB, GENERATION,
                         stats=stats_fit, halloffame=hof, verbose=verbose)
    train_time = time.time() - t0

    best_ind = hof[0]
    test_acc, clf_name, train_feat, test_feat = eval_test(
        toolbox, best_ind, x_train, y_train, x_test, y_test,
    )

    if verbose:
        print(f'\n=== GP-FR Results (seed={seed}) ===')
        print(f'Training time: {train_time:.1f}s')
        print(f'Best train fitness: {best_ind.fitness.values[0]:.2f}%')
        print(f'Test accuracy: {test_acc:.2f}% (classifier: {clf_name})')
        print(f'Feature dim: {train_feat.shape[1]}')
        print(f'Program: {best_ind}')

    return {
        'test_acc': test_acc,
        'classifier': clf_name,
        'train_time': train_time,
        'train_fitness': best_ind.fitness.values[0],
        'feature_dim': train_feat.shape[1],
        'program': str(best_ind),
        'train_features': train_feat,
        'test_features': test_feat,
    }


def main():
    parser = argparse.ArgumentParser(description='GP-FR baseline')
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--train_label', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--test_label', required=True)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    x_train = np.load(args.train_data) / 255.0
    y_train = np.load(args.train_label)
    x_test = np.load(args.test_data) / 255.0
    y_test = np.load(args.test_label)

    print(f'Train: {x_train.shape}, Test: {x_test.shape}')

    results = []
    for run in range(args.runs):
        seed = args.seed + run
        print(f'\n--- Run {run + 1}/{args.runs} (seed={seed}) ---')
        res = run_gp_fr(x_train, y_train, x_test, y_test, seed=seed)
        results.append(res)

    accs = [r['test_acc'] for r in results]
    print(f'\n=== Summary over {len(results)} runs ===')
    print(f'Mean ± Std: {np.mean(accs):.2f} ± {np.std(accs):.2f}')
    print(f'Max: {np.max(accs):.2f}')


if __name__ == '__main__':
    main()
