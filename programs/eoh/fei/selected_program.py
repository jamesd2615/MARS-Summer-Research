import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    centre = image[
        int(0.10 * h):int(0.90 * h),
        int(0.30 * w):int(0.70 * w)
    ]

    dx = np.diff(image, axis=0)
    dy = np.diff(image, axis=1)

    quarter = w // 4
    left = image[:, :quarter]
    right = np.fliplr(image[:, w-quarter:])

    threshold = np.median(image) + 0.5 * np.std(image)

    return np.array(
        [
            np.mean(image),
            np.var(image),
            np.mean(centre),
            np.var(centre),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(np.abs(left - right)),
            np.mean(image > threshold),
        ],
        dtype=float,
    )
