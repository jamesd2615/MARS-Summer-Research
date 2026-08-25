import numpy as np


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Simple 8-feature test program for the modular EOH pipeline.

    This is only a smoke-test program.
    It is not an EOH-discovered research result.
    """
    image = np.asarray(image, dtype=float)

    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    return np.array(
        [
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image),
            np.median(image),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(image > np.mean(image)),
        ],
        dtype=float,
    )