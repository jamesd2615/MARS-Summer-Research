# Selected EOH programs

This directory contains the 40 EOH programs selected during the final nested
cross-validation experiments:

```text
<dataset>/seed_<42|43>/fold_<1..5>/selected_program.py
```

Each program was selected using only an internal split of its outer training
fold, then evaluated on the held-out outer validation fold. Identical files in
different fold directories are retained deliberately because recurrence is part
of the explainability analysis.

Corresponding best-sample metadata is under
`results/search_provenance/eoh/`.
