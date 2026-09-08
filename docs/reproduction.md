# Reproduction guide

## Environment

The final experiments used Python 3.13 on Windows with CPU-based PyTorch.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e vendor/eoh/eoh
```

Activate the environment using the command appropriate to the operating system,
then follow `data/README.md` to install the datasets locally.

## Experiment commands

Run commands from the repository root. Available datasets are `fei`, `ksdd2`,
`mvtec`, and `stl10`; methods are `gp_original`, `gp_modified`, `eoh`, `cnn`,
and `mlp`.

Final GP configuration example:

```bash
python -m src.experiments.run_experiment \
  --dataset fei --method gp_original --folds 5 --seed 42 \
  --population 10 --generations 2
```

Final nested EOH configuration example:

```bash
python -m src.experiments.run_experiment \
  --dataset fei --method eoh --folds 5 --seed 42 \
  --eoh-population 5 --eoh-generations 5 --eoh-validation-size 0.20
```

Fresh EOH search requires `DEEPSEEK_API_KEY` in the environment. The outer
validation folds are not supplied to program selection.

Evaluate a retained program without an API request:

```bash
python -m src.experiments.run_experiment \
  --dataset fei --method eoh --folds 5 --seed 42 \
  --eoh-program-file programs/eoh/fei/seed_42/fold_1/selected_program.py
```

Neural example:

```bash
python -m src.experiments.run_experiment \
  --dataset stl10 --method cnn --folds 5 --seed 42 \
  --epochs 10 --batch-size 64 --learning-rate 0.001
```

## Analyses

The committed results can be re-analysed without downloading image datasets:

```bash
python -m src.analysis.statistical_analysis
python -m src.analysis.explainability_analysis
```

Run the fast integrity suite with:

```bash
python -m unittest discover -s tests -v
```

Run `notebooks/analysis/poster_visualisations.ipynb` to recreate the committed
figure assets. Graphviz must be installed as a system executable as well as via
the Python package.

## Generated material

EOH population histories, diagnostics, caches, checkpoints, local datasets, and
model weights are ignored by Git. Compact best-sample provenance and the final
selected programs are committed so the published analysis remains inspectable.
