# Experimental design

## Compared methods

The study compares GP Original, GP Modified, EOH, CNN, and MLP. GP Original and
GP Modified evolve explicit image-processing trees. EOH uses an LLM-guided
evolutionary process to produce an executable eight-feature Python function.
CNN and MLP provide learned neural baselines.

For GP and EOH, the extracted representation is standardised and evaluated by
a linear SVM. The CNN and MLP learn representations and classifiers end to end.

## Cross-validation and seeds

Every method is evaluated on the same five stratified outer folds for each
dataset. Seeds 42 and 43 define the common paired inferential design. CNN and
MLP were also run with seed 44; those observations are reported descriptively
but are excluded from direct five-method hypothesis tests.

## Leakage-safe EOH selection

EOH search occurs separately inside each outer training fold. That training
fold is split into an internal stratified 80% search-training subset and 20%
search-validation subset. Candidate programs are refined and selected using
only these internal subsets. The chosen program is then applied to the unseen
outer validation fold.

The repository retains 40 programs: four datasets × two seeds × five folds.

## Metrics

Validation macro F1 is the primary metric for all datasets. Accuracy is a
secondary metric for FEI and STL-10; balanced accuracy is secondary for the
imbalanced KSDD2 and MVTec tasks.

## Statistical inference

Pairwise method differences use the corrected repeated-cross-validation paired
t-test indexed by `(experiment_seed, fold)`. With ten paired observations, the
variance correction is `1/n + 1/4`, and tests use nine degrees of freedom.
Two-sided p-values are adjusted by the Holm procedure within each dataset and
metric family.
