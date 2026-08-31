from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["FEI", "KSDD2", "MVTEC", "STL10"]
GP_METHODS = ["gp_original", "gp_modified"]


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _normalise_dataset(name: str) -> str:
    mapping = {
        "fei": "FEI",
        "ksdd2": "KSDD2",
        "mvtec": "MVTEC",
        "stl10": "STL10",
    }
    return mapping.get(str(name).lower(), str(name).upper())


def _max_parenthesis_depth(text: str) -> int:
    depth = 0
    max_depth = 0
    for char in text:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(depth - 1, 0)
    return max_depth


def _function_tokens(text: str) -> list[str]:
    return re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)


def _safe_mean(values):
    values = list(values)
    return float(np.mean(values)) if values else np.nan


# ---------------------------------------------------------------------
# GP analysis
# ---------------------------------------------------------------------

GP_FAMILIES = {
    "gradient_edge": {
        "HOG", "Sobel", "Sobel_R", "SobelX", "SobelX_R",
        "SobelY", "SobelY_R", "Lap", "Lap_R", "LoG1", "LoG1_R",
        "LoG2", "LoG2_R", "GauD", "GauD_R", "DIF", "DIF_R",
    },
    "texture_local_descriptor": {
        "uLBP", "uLBP_R", "SIFT", "SIFT_R",
    },
    "intensity_statistics": {
        "Hist", "Hist_R", "Mean", "Mean_R", "Med", "Med_R",
        "Min", "Min_R", "Max", "Max_R", "RIF", "RIF_R",
    },
    "smoothing_filtering": {
        "Gau", "Gau_R",
    },
    "spatial_region": {
        "RegionR", "RegionS",
    },
}


def _gp_family_presence(tokens: list[str]) -> dict[str, bool]:
    token_set = set(tokens)
    return {
        family: bool(token_set.intersection(operators))
        for family, operators in GP_FAMILIES.items()
    }


def load_gp_programs(results_root: str | Path = "results/final") -> pd.DataFrame:
    """
    Load the final GP programs from seed_42 and seed_43 result folders.

    The experiment repeat is taken from the directory name. The CSV's
    existing 'seed' field is preserved as the internal/fold seed.
    """
    root = Path(results_root)
    rows = []

    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue

        for method in GP_METHODS:
            method_dir = dataset_dir / method
            if not method_dir.exists():
                continue

            for experiment_seed in (42, 43):
                csv_path = method_dir / f"seed_{experiment_seed}" / "fold_results.csv"
                if not csv_path.exists():
                    continue

                df = pd.read_csv(csv_path)
                if "program" not in df.columns:
                    continue

                for _, row in df.iterrows():
                    program = str(row["program"])
                    tokens = _function_tokens(program)
                    token_counts = Counter(tokens)
                    family_presence = _gp_family_presence(tokens)

                    top_match = re.match(r"\s*(FC[234])\s*\(", program)
                    top_operator = top_match.group(1) if top_match else ""

                    rec = {
                        "dataset": _normalise_dataset(row.get("dataset", dataset_dir.name)),
                        "method": method,
                        "experiment_seed": experiment_seed,
                        "fold": int(row["fold"]),
                        "internal_seed": row.get("seed", np.nan),
                        "validation_macro_f1": row.get("validation_macro_f1", np.nan),
                        "program": program,
                        "program_length_chars": len(program),
                        "function_call_count": len(tokens),
                        "max_parenthesis_depth": _max_parenthesis_depth(program),
                        "unique_operator_count": len(set(tokens)),
                        "top_feature_combiner": top_operator,
                        "image_terminal_count": len(re.findall(r"\bImage\b", program)),
                    }

                    for family, present in family_presence.items():
                        rec[f"family_{family}"] = int(present)

                    for token, count in token_counts.items():
                        rec[f"op__{token}"] = count

                    rows.append(rec)

    gp = pd.DataFrame(rows)

    if gp.empty:
        raise FileNotFoundError(
            "No final GP fold_results.csv files with a 'program' column were found."
        )

    expected = len(DATASETS) * len(GP_METHODS) * 2 * 5
    if len(gp) != expected:
        print(
            f"WARNING: expected {expected} final GP programs "
            f"(4 datasets x 2 methods x 2 seeds x 5 folds), found {len(gp)}."
        )

    return gp.sort_values(
        ["dataset", "method", "experiment_seed", "fold"]
    ).reset_index(drop=True)


def summarise_gp_complexity(gp: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "program_length_chars",
        "function_call_count",
        "max_parenthesis_depth",
        "unique_operator_count",
        "image_terminal_count",
    ]

    rows = []
    for (dataset, method), group in gp.groupby(["dataset", "method"]):
        rec = {
            "dataset": dataset,
            "method": method,
            "n_programs": len(group),
        }
        for metric in metrics:
            rec[f"{metric}_mean"] = group[metric].mean()
            rec[f"{metric}_std"] = group[metric].std()
            rec[f"{metric}_median"] = group[metric].median()
        rows.append(rec)

    return pd.DataFrame(rows)


def summarise_gp_families(gp: pd.DataFrame) -> pd.DataFrame:
    family_cols = [c for c in gp.columns if c.startswith("family_")]
    rows = []

    for (dataset, method), group in gp.groupby(["dataset", "method"]):
        for col in family_cols:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "feature_family": col.replace("family_", ""),
                    "programs_using_family": int(group[col].sum()),
                    "n_programs": len(group),
                    "proportion_programs": group[col].mean(),
                }
            )

    return pd.DataFrame(rows)


def summarise_gp_operators(gp: pd.DataFrame) -> pd.DataFrame:
    op_cols = [c for c in gp.columns if c.startswith("op__")]
    rows = []

    for (dataset, method), group in gp.groupby(["dataset", "method"]):
        for col in op_cols:
            counts = group[col] if col in group else 0
            total = float(counts.fillna(0).sum())
            programs = int((counts.fillna(0) > 0).sum())
            if total > 0:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "operator": col.replace("op__", ""),
                        "total_occurrences": int(total),
                        "programs_using_operator": programs,
                        "proportion_programs": programs / len(group),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["dataset", "method", "programs_using_operator", "total_occurrences"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# EOH analysis
# ---------------------------------------------------------------------

EOH_FAMILY_PATTERNS = {
    "colour": [
        r"\bred\b", r"\bgreen\b", r"\bblue\b", r"\brgb\b",
        r"colou?r", r"channel", r"saturation", r"\bhue\b",
    ],
    "gradient_edge": [
        r"gradient", r"\bgrad_", r"\bdx\b", r"\bdy\b", r"sobel",
        r"edge", r"np\.diff", r"laplac", r"derivative",
    ],
    "intensity_statistics": [
        r"np\.mean", r"np\.std", r"np\.median", r"percentile",
        r"quantile", r"variance", r"\bskew", r"threshold",
        r"histogram", r"\bintensity\b",
    ],
    "spatial_region_symmetry": [
        r"centre", r"center", r"\bleft\b", r"\bright\b", r"\btop\b",
        r"\bbottom\b", r"region", r"\bband\b", r"quadrant",
        r"symmetr", r"asym", r"mask", r"triangle", r"outer",
    ],
    "local_texture_structure": [
        r"\blocal\b", r"sliding_window", r"\bpatch", r"entropy",
        r"\blbp\b", r"binary", r"texture", r"window",
    ],
    "frequency_transform": [
        r"\bfft\b", r"fourier", r"frequency", r"spectrum",
    ],
}


def _canonical_python(source: str) -> str:
    """
    Canonicalise Python source for recurrence counting.

    Comments and formatting differences are removed via AST unparse where
    possible, so 'same program' means the same parsed Python structure.
    """
    try:
        tree = ast.parse(source)
        return ast.unparse(tree)
    except Exception:
        return "\n".join(
            line.strip()
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )


def _extract_returned_feature_count(source: str) -> int | float:
    """
    Best-effort count of scalar expressions returned in np.array([...]).
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return np.nan

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            value = node.value
            if isinstance(value, ast.Call):
                func = value.func
                is_array = (
                    isinstance(func, ast.Attribute)
                    and func.attr == "array"
                )
                if is_array and value.args:
                    first = value.args[0]
                    if isinstance(first, (ast.List, ast.Tuple)):
                        return len(first.elts)
    return np.nan


def _eoh_family_presence(source: str) -> dict[str, bool]:
    lower = source.lower()
    return {
        family: any(re.search(pattern, lower) for pattern in patterns)
        for family, patterns in EOH_FAMILY_PATTERNS.items()
    }


def load_eoh_programs(programs_root: str | Path = "programs/eoh") -> pd.DataFrame:
    """
    Load the 40 final nested-CV EOH selected programs.

    Expected structure:
      programs/eoh/<dataset>/seed_<42|43>/fold_<1..5>/selected_program.py
    """
    root = Path(programs_root)
    rows = []

    for dataset in ["fei", "ksdd2", "mvtec", "stl10"]:
        for experiment_seed in (42, 43):
            for fold in range(1, 6):
                path = (
                    root
                    / dataset
                    / f"seed_{experiment_seed}"
                    / f"fold_{fold}"
                    / "selected_program.py"
                )
                if not path.exists():
                    continue

                source = path.read_text(encoding="utf-8")
                canonical = _canonical_python(source)
                family_presence = _eoh_family_presence(source)
                tokens = _function_tokens(source)

                rec = {
                    "dataset": _normalise_dataset(dataset),
                    "method": "eoh",
                    "experiment_seed": experiment_seed,
                    "fold": fold,
                    "path": str(path),
                    "source": source,
                    "canonical_hash": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()[:16],
                    "source_length_chars": len(source),
                    "source_line_count": len(source.splitlines()),
                    "function_call_count": len(tokens),
                    "returned_feature_count": _extract_returned_feature_count(source),
                }

                for family, present in family_presence.items():
                    rec[f"family_{family}"] = int(present)

                rows.append(rec)

    eoh = pd.DataFrame(rows)

    if eoh.empty:
        raise FileNotFoundError(
            "No final nested EOH selected_program.py files were found."
        )

    if len(eoh) != 40:
        print(
            "WARNING: expected 40 final EOH programs "
            f"(4 datasets x 2 seeds x 5 folds), found {len(eoh)}."
        )

    return eoh.sort_values(
        ["dataset", "experiment_seed", "fold"]
    ).reset_index(drop=True)


def summarise_eoh_stability(eoh: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, group in eoh.groupby("dataset"):
        counts = group["canonical_hash"].value_counts()
        n = len(group)
        n_unique = counts.size
        max_recurrence = int(counts.iloc[0])

        rows.append(
            {
                "dataset": dataset,
                "n_programs": n,
                "unique_programs": n_unique,
                "duplicate_selections": n - n_unique,
                "most_common_program_recurrence": max_recurrence,
                "most_common_program_proportion": max_recurrence / n,
            }
        )

    return pd.DataFrame(rows)


def summarise_eoh_families(eoh: pd.DataFrame) -> pd.DataFrame:
    family_cols = [c for c in eoh.columns if c.startswith("family_")]
    rows = []

    for dataset, group in eoh.groupby("dataset"):
        for col in family_cols:
            rows.append(
                {
                    "dataset": dataset,
                    "method": "eoh",
                    "feature_family": col.replace("family_", ""),
                    "programs_using_family": int(group[col].sum()),
                    "n_programs": len(group),
                    "proportion_programs": group[col].mean(),
                }
            )

    return pd.DataFrame(rows)


def summarise_eoh_complexity(eoh: pd.DataFrame) -> pd.DataFrame:
    return (
        eoh.groupby("dataset")
        .agg(
            n_programs=("canonical_hash", "size"),
            source_length_chars_mean=("source_length_chars", "mean"),
            source_length_chars_std=("source_length_chars", "std"),
            source_line_count_mean=("source_line_count", "mean"),
            function_call_count_mean=("function_call_count", "mean"),
            returned_feature_count_mean=("returned_feature_count", "mean"),
            returned_feature_count_min=("returned_feature_count", "min"),
            returned_feature_count_max=("returned_feature_count", "max"),
        )
        .reset_index()
        .assign(method="eoh")
    )


def eoh_recurrence_table(eoh: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, group in eoh.groupby("dataset"):
        for program_hash, pg in group.groupby("canonical_hash"):
            selections = [
                f"seed_{int(r.experiment_seed)}/fold_{int(r.fold)}"
                for r in pg.itertuples()
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "canonical_hash": program_hash,
                    "selection_count": len(pg),
                    "selection_locations": "; ".join(selections),
                    "representative_path": pg.iloc[0]["path"],
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["dataset", "selection_count"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Combined Section 5 evidence
# ---------------------------------------------------------------------

def combined_family_table(
    gp_families: pd.DataFrame,
    eoh_families: pd.DataFrame,
) -> pd.DataFrame:
    return (
        pd.concat([gp_families, eoh_families], ignore_index=True)
        .sort_values(["dataset", "method", "feature_family"])
        .reset_index(drop=True)
    )


def section5_headline_summary(
    eoh_stability: pd.DataFrame,
    gp_complexity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact evidence table useful when drafting the final poster section.
    """
    gp_small = gp_complexity[
        [
            "dataset",
            "method",
            "function_call_count_mean",
            "max_parenthesis_depth_mean",
            "unique_operator_count_mean",
        ]
    ].copy()

    stability = eoh_stability.copy()
    stability["method"] = "eoh"

    return stability.merge(
        gp_small.groupby("dataset").agg(
            gp_function_calls_mean=("function_call_count_mean", "mean"),
            gp_parenthesis_depth_mean=("max_parenthesis_depth_mean", "mean"),
            gp_unique_operators_mean=("unique_operator_count_mean", "mean"),
        ).reset_index(),
        on="dataset",
        how="left",
    )


def run_section5_analysis(
    results_root: str | Path = "results/final",
    programs_root: str | Path = "programs/eoh",
    output_dir: str | Path = "results/explainability_analysis",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running MARS Section 5 explainability analysis...")
    print("=" * 62)

    print("[1/7] Loading final GP programs....................", end=" ")
    gp = load_gp_programs(results_root)
    gp.to_csv(output_dir / "gp_programs.csv", index=False)
    print(f"DONE ({len(gp)} programs)")

    print("[2/7] GP complexity.................................", end=" ")
    gp_complexity = summarise_gp_complexity(gp)
    gp_complexity.to_csv(output_dir / "gp_complexity_summary.csv", index=False)
    print("DONE")

    print("[3/7] GP operators and feature families.............", end=" ")
    gp_families = summarise_gp_families(gp)
    gp_operators = summarise_gp_operators(gp)
    gp_families.to_csv(output_dir / "gp_feature_families.csv", index=False)
    gp_operators.to_csv(output_dir / "gp_operator_frequencies.csv", index=False)
    print("DONE")

    print("[4/7] Loading final EOH programs....................", end=" ")
    eoh = load_eoh_programs(programs_root)
    eoh.drop(columns=["source"]).to_csv(
        output_dir / "eoh_program_index.csv", index=False
    )
    print(f"DONE ({len(eoh)} programs)")

    print("[5/7] EOH recurrence/stability......................", end=" ")
    eoh_stability = summarise_eoh_stability(eoh)
    eoh_recurrence = eoh_recurrence_table(eoh)
    eoh_stability.to_csv(output_dir / "eoh_stability_summary.csv", index=False)
    eoh_recurrence.to_csv(output_dir / "eoh_program_recurrence.csv", index=False)
    print("DONE")

    print("[6/7] EOH feature families and complexity...........", end=" ")
    eoh_families = summarise_eoh_families(eoh)
    eoh_complexity = summarise_eoh_complexity(eoh)
    eoh_families.to_csv(output_dir / "eoh_feature_families.csv", index=False)
    eoh_complexity.to_csv(output_dir / "eoh_complexity_summary.csv", index=False)
    print("DONE")

    print("[7/7] Combined Section 5 evidence....................", end=" ")
    combined = combined_family_table(gp_families, eoh_families)
    headline = section5_headline_summary(eoh_stability, gp_complexity)
    combined.to_csv(output_dir / "combined_feature_families.csv", index=False)
    headline.to_csv(output_dir / "section5_headline_summary.csv", index=False)
    print("DONE")

    print("=" * 62)
    print("Explainability analysis complete.")
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("EOH stability:")
    for row in eoh_stability.itertuples():
        print(
            f"  {row.dataset}: {row.unique_programs}/{row.n_programs} unique; "
            f"most common selected {row.most_common_program_recurrence} time(s)"
        )

    print()
    print("Mean GP complexity (function calls / nesting depth):")
    for row in gp_complexity.itertuples():
        print(
            f"  {row.dataset:5s} {row.method:11s}: "
            f"{row.function_call_count_mean:.1f} / "
            f"{row.max_parenthesis_depth_mean:.1f}"
        )

    print()
    print("Files:")
    for path in sorted(output_dir.glob("*.csv")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    run_section5_analysis()
