import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")

    h, w, _ = image.shape

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    intensity = np.mean(image, axis=2)

    centre = intensity[
        int(0.20 * h):int(0.85 * h),
        int(0.20 * w):int(0.80 * w)
    ]

    dx = np.diff(intensity, axis=1)
    dy = np.diff(intensity, axis=0)
    colour_range = np.max(image, axis=2) - np.min(image, axis=2)

    return np.array(
        [
            np.mean(red),
            np.mean(green),
            np.mean(blue),
            np.std(intensity),
            np.mean(centre) - np.mean(intensity),
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            np.mean(colour_range),
        ],
        dtype=float,
    )
