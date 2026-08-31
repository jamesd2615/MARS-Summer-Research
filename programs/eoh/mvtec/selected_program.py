import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    top = image[:int(0.5*h), :]
    bottom = image[int(0.5*h):, :]

    centre = image[
        int(0.25*h):int(0.75*h),
        int(0.25*w):int(0.75*w)
    ]

    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    # diagonal gradients
    d1 = np.abs(np.diff(image, axis=0)[:, :-1])
    d2 = np.abs(np.diff(image, axis=0)[:, 1:])

    half_h = h // 2
    top_half = image[:half_h, :]
    bottom_half = np.flipud(image[h-half_h:, :])

    threshold = np.mean(image) - np.std(image)

    return np.array([
        np.mean(top),
        np.mean(bottom),
        np.mean(np.abs(centre - np.mean(centre))),
        np.mean(np.abs(dx)),
        np.mean(np.abs(dy)),
        np.mean(d1 - d2),
        np.mean(np.abs(top_half - bottom_half)),
        np.mean(image < threshold)
    ], dtype=float)
