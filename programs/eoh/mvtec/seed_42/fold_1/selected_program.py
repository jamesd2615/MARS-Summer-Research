import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    centre = image[
        int(0.20 * h):int(0.85 * h),
        int(0.20 * w):int(0.80 * w)
    ]

    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    half = w // 2
    left = image[:, :half]
    right = np.fliplr(image[:, w-half:])

    threshold = np.mean(image) + np.std(image)

    return np.array(
        [
            np.mean(image),
            np.std(image),
            np.mean(centre),
            np.std(centre),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(np.abs(left - right)),
            np.mean(image > threshold),
        ],
        dtype=float,
    )
