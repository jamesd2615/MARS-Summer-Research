from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_linear_svm(
    *,
    c: float = 1.0,
    random_state: int = 42,
    max_iter: int = 10000,
):
    """
    Build the standard classifier used across MARS feature
    extraction experiments.

    The pipeline standardises extracted features and then
    fits a linear Support Vector Machine.

    Parameters
    ----------
    c:
        LinearSVC regularisation parameter.

    random_state:
        Random seed used by LinearSVC.

    max_iter:
        Maximum number of optimisation iterations.

    Returns
    -------
    sklearn.pipeline.Pipeline
        StandardScaler -> LinearSVC
    """
    if c <= 0:
        raise ValueError(
            "c must be greater than zero."
        )

    if max_iter <= 0:
        raise ValueError(
            "max_iter must be greater than zero."
        )

    return make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=c,
            random_state=random_state,
            dual=True,
            max_iter=max_iter,
        ),
    )
