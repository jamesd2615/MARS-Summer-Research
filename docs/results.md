# Results guide

## Dataset-level findings

| Dataset | Numerical macro-F1 winner | Interpretation |
|---|---|---|
| FEI | GP Original | Strongest mean, but no pairwise difference survived Holm adjustment |
| KSDD2 | CNN | MLP accuracy concealed near-majority-class behaviour |
| MVTec AD subset | GP Modified | Explainable GP was strongest numerically; EOH was weaker |
| STL-10 | CNN | CNN clearly led several methods; GP Original remained statistically competitive |

The cross-dataset conclusion is that representation quality depends on task
structure. Interpretability and recurrence do not guarantee adequacy: EOH
repeatedly rediscovered a compact STL-10 program, but its predictive performance
remained below CNN and GP Original.

## Definitive tables

| File | Purpose |
|---|---|
| `results/final/final_master_folds.csv` | All 240 completed fold observations |
| `results/final/final_paired_folds.csv` | The 200 common-seed observations used for inference |
| `results/final/final_master_summary.csv` | Dataset/method means and standard deviations |
| `results/final/final_primary_pairwise_statistics.csv` | Primary macro-F1 tests |
| `results/statistical_analysis/` | Full primary, secondary, robustness, and gap analyses |
| `results/explainability_analysis/` | GP operators, EOH features, complexity, and recurrence |

## Reading the imbalanced results

On KSDD2, the MLP achieved mean validation accuracy of approximately 0.893 but
balanced accuracy of approximately 0.501 and macro F1 of approximately 0.473.
The latter metrics reveal that the apparently strong accuracy largely reflects
the majority class.
