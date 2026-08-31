from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests


PRIMARY_METRIC = "validation_macro_f1"

SECONDARY_METRICS = {
    "FEI": "validation_accuracy",
    "KSDD2": "validation_balanced_accuracy",
    "MVTEC": "validation_balanced_accuracy",
    "STL10": "validation_accuracy",
}

DATASETS = ["FEI", "KSDD2", "MVTEC", "STL10"]
METHODS = ["gp_original", "gp_modified", "eoh", "cnn", "mlp"]


def corrected_repeated_cv_ttest(
    scores_a: pd.Series,
    scores_b: pd.Series,
    test_train_ratio: float = 1 / 4,
) -> dict:
    """
    Corrected repeated k-fold CV t-test for paired model scores.

    For the MARS final experiment:
    - 5-fold CV => n_test / n_train ~= 0.25
    - formal five-method inference uses experiment seeds 42 and 43
    - n = 10 paired fold observations per method comparison
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired score vectors must have equal length.")

    differences = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(differences)

    if n < 2:
        raise ValueError("At least two paired observations are required.")

    mean_difference = differences.mean()
    sd_difference = differences.std(ddof=1)

    correction_factor = (1 / n) + test_train_ratio
    corrected_se = np.sqrt(correction_factor * sd_difference**2)

    if corrected_se == 0:
        t_statistic = np.inf if mean_difference != 0 else 0.0
        p_value = 0.0 if mean_difference != 0 else 1.0
    else:
        t_statistic = mean_difference / corrected_se
        p_value = 2 * tdist.sf(abs(t_statistic), df=n - 1)

    cohens_dz = (
        mean_difference / sd_difference
        if sd_difference != 0
        else np.nan
    )

    return {
        "n_pairs": n,
        "mean_difference": mean_difference,
        "sd_difference": sd_difference,
        "cohens_dz": cohens_dz,
        "correction_factor": correction_factor,
        "corrected_se": corrected_se,
        "t_statistic": t_statistic,
        "df": n - 1,
        "p_value": p_value,
    }


def pairwise_dataset_comparisons(
    df: pd.DataFrame,
    dataset: str,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """
    Run all 10 pairwise corrected repeated-CV comparisons for one dataset.

    Methods are paired on identical experiment_seed/fold blocks.
    Holm correction is applied within the 10 comparisons for that
    dataset/metric family.
    """
    subset = df[df["dataset"] == dataset].copy()

    if subset.empty:
        raise ValueError(f"No rows found for dataset {dataset!r}.")

    pivot = subset.pivot(
        index=["experiment_seed", "fold"],
        columns="method",
        values=metric,
    )

    missing_methods = [m for m in METHODS if m not in pivot.columns]
    if missing_methods:
        raise ValueError(
            f"{dataset}: missing methods for {metric}: {missing_methods}"
        )

    pivot = pivot[METHODS]

    if pivot.isna().any().any():
        bad_rows = pivot[pivot.isna().any(axis=1)]
        raise ValueError(
            f"{dataset}: incomplete paired blocks for {metric}:\n{bad_rows}"
        )

    rows = []

    for method_a, method_b in combinations(METHODS, 2):
        result = corrected_repeated_cv_ttest(
            pivot[method_a],
            pivot[method_b],
        )

        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                **result,
            }
        )

    results = pd.DataFrame(rows)

    _, p_adjusted, _, _ = multipletests(
        results["p_value"],
        alpha=0.05,
        method="holm",
    )

    results["p_value_holm"] = p_adjusted
    results["significant_holm_05"] = results["p_value_holm"] < 0.05

    return results


def validate_paired_input(df: pd.DataFrame) -> None:
    """
    Validate the formal five-method inference dataset.
    """
    required = {
        "dataset",
        "method",
        "experiment_seed",
        "fold",
        "train_macro_f1",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"final_paired_folds.csv is missing required columns: {sorted(missing)}"
        )

    unexpected_seeds = sorted(set(df["experiment_seed"]) - {42, 43})
    if unexpected_seeds:
        raise ValueError(
            "Formal paired inference must use experiment seeds 42 and 43 only. "
            f"Unexpected seeds found: {unexpected_seeds}"
        )

    duplicates = df.duplicated(
        subset=["dataset", "experiment_seed", "fold", "method"]
    ).sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicated paired rows.")

    expected_rows = len(DATASETS) * len(METHODS) * 2 * 5
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} paired rows, found {len(df)}."
        )

    counts = (
        df.groupby(["dataset", "method", "experiment_seed"])
        .size()
    )
    if not (counts == 5).all():
        raise ValueError(
            "Every dataset/method/experiment_seed group must contain exactly 5 folds."
        )


def descriptive_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive validation performance for the common-seed paired data.
    """
    metrics = [
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
    ]

    summary = (
        df.groupby(["dataset", "method"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    return summary


def generalisation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise train/validation macro-F1 and mean generalisation gap.
    """
    temp = df.copy()
    temp["generalisation_gap"] = (
        temp["train_macro_f1"] - temp["validation_macro_f1"]
    )

    return (
        temp.groupby(["dataset", "method"])
        .agg(
            train_macro_f1_mean=("train_macro_f1", "mean"),
            train_macro_f1_std=("train_macro_f1", "std"),
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_std=("validation_macro_f1", "std"),
            generalisation_gap_mean=("generalisation_gap", "mean"),
            generalisation_gap_std=("generalisation_gap", "std"),
        )
        .reset_index()
    )


def primary_pairwise_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary inferential analysis: validation macro-F1 for all datasets.
    """
    frames = [
        pairwise_dataset_comparisons(
            df=df,
            dataset=dataset,
            metric=PRIMARY_METRIC,
        )
        for dataset in DATASETS
    ]

    return pd.concat(frames, ignore_index=True)


def secondary_metric_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-specified secondary validation metrics.

    FEI/STL10:
        validation accuracy
    KSDD2/MVTEC:
        validation balanced accuracy
    """
    frames = []

    for dataset, metric in SECONDARY_METRICS.items():
        frames.append(
            pairwise_dataset_comparisons(
                df=df,
                dataset=dataset,
                metric=metric,
            )
        )

    return pd.concat(frames, ignore_index=True)


def neural_seed44_robustness(
    master_file: str | Path = "results/final/final_master_folds.csv",
) -> pd.DataFrame:
    """
    Descriptive CNN/MLP robustness summary including experiment seed 44.

    Seed 44 is NOT used in formal five-method inference because GP/EOH
    only share experiment seeds 42 and 43. This table is descriptive only.
    """
    path = Path(master_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find neural robustness input: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "dataset",
        "method",
        "experiment_seed",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    neural = df[
        df["method"].isin(["cnn", "mlp"])
        & df["experiment_seed"].isin([42, 43, 44])
    ].copy()

    summary = (
        neural.groupby(["dataset", "method", "experiment_seed"])
        .agg(
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_std=("validation_accuracy", "std"),
            validation_balanced_accuracy_mean=(
                "validation_balanced_accuracy",
                "mean",
            ),
            validation_balanced_accuracy_std=(
                "validation_balanced_accuracy",
                "std",
            ),
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_std=("validation_macro_f1", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
    )

    return summary


def headline_summary(
    descriptive: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact dataset-level table for poster/report drafting.
    """
    mean_col = "validation_macro_f1_mean"

    ranked = descriptive[
        ["dataset", "method", mean_col]
    ].copy()

    ranked["rank"] = ranked.groupby("dataset")[mean_col].rank(
        ascending=False,
        method="min",
    )

    winners = (
        ranked[ranked["rank"] == 1]
        .sort_values(["dataset", "method"])
        .rename(
            columns={
                "method": "best_method_macro_f1",
                mean_col: "best_validation_macro_f1",
            }
        )
        [
            [
                "dataset",
                "best_method_macro_f1",
                "best_validation_macro_f1",
            ]
        ]
    )

    significant_counts = (
        primary.groupby("dataset")["significant_holm_05"]
        .sum()
        .astype(int)
        .rename("n_significant_primary_pairwise")
        .reset_index()
    )

    return winners.merge(significant_counts, on="dataset", how="left")


def run_full_analysis(
    paired_file: str | Path = "results/final/final_paired_folds.csv",
    master_file: str | Path = "results/final/final_master_folds.csv",
    output_dir: str | Path = "results/statistical_analysis",
) -> None:
    """
    Run the complete final MARS statistical analysis pipeline.
    """
    paired_file = Path(paired_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not paired_file.exists():
        raise FileNotFoundError(
            f"Could not find paired inference file: {paired_file}"
        )

    print("Running MARS final statistical analysis...")
    print("=" * 58)

    paired = pd.read_csv(paired_file)
    validate_paired_input(paired)

    print("[1/6] Descriptive performance....................", end=" ")
    descriptive = descriptive_performance(paired)
    descriptive.to_csv(
        output_dir / "descriptive_performance.csv",
        index=False,
    )
    print("DONE")

    print("[2/6] Primary macro-F1 inference................", end=" ")
    primary = primary_pairwise_analysis(paired)
    primary.to_csv(
        output_dir / "primary_macro_f1_pairwise.csv",
        index=False,
    )
    # Preserve the existing top-level final output used earlier.
    primary.to_csv(
        "results/final/final_primary_pairwise_statistics.csv",
        index=False,
    )
    print("DONE")

    print("[3/6] Secondary metric validation...............", end=" ")
    secondary = secondary_metric_analysis(paired)
    secondary.to_csv(
        output_dir / "secondary_metric_pairwise.csv",
        index=False,
    )
    print("DONE")

    print("[4/6] Generalisation analysis...................", end=" ")
    gaps = generalisation_summary(paired)
    gaps.to_csv(
        output_dir / "generalisation_macro_f1.csv",
        index=False,
    )
    print("DONE")

    print("[5/6] Neural seed-44 robustness.................", end=" ")
    robustness = neural_seed44_robustness(master_file)
    robustness.to_csv(
        output_dir / "neural_seed44_robustness.csv",
        index=False,
    )
    print("DONE")

    print("[6/6] Compact headline summary..................", end=" ")
    headline = headline_summary(descriptive, primary)
    headline.to_csv(
        output_dir / "headline_summary.csv",
        index=False,
    )
    print("DONE")

    print("=" * 58)
    print("Statistical analysis complete.")
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Files:")
    for path in sorted(output_dir.glob("*.csv")):
        print(f"  - {path.name}")

    print()
    print("Primary Holm-significant comparisons by dataset:")
    counts = (
        primary.groupby("dataset")["significant_holm_05"]
        .sum()
        .astype(int)
    )
    for dataset in DATASETS:
        print(f"  {dataset}: {counts.get(dataset, 0)} / 10")

    print()
    print(
        "Note: formal five-method inference uses experiment seeds 42/43. "
        "CNN/MLP seed 44 is descriptive robustness evidence only."
    )


if __name__ == "__main__":
    run_full_analysis()
