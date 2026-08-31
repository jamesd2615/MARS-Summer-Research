import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # central region with different bounds
    centre = image[
        int(0.10 * h):int(0.90 * h),
        int(0.30 * w):int(0.70 * w)
    ]

    # gradients
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    # symmetry: compare top-bottom instead of left-right
    half_h = h // 2
    top = image[:half_h, :]
    bottom = np.flipud(image[h - half_h:, :])

    # threshold using median and 0.5*std
    threshold = np.median(image) + 0.5 * np.std(image)

    # central moments (second order)
    cy, cx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx = cx - np.mean(cx)
    cy = cy - np.mean(cy)
    mu20 = np.mean(image * (cx ** 2)) / (w**2)
    mu02 = np.mean(image * (cy ** 2)) / (h**2)

    return np.array([
        np.mean(image),
        np.median(image),
        np.mean(centre),
        np.std(centre),
        np.mean(np.abs(dx)),
        np.mean(np.abs(dy)),
        np.mean(np.abs(top - bottom)),
        np.mean(image > threshold) + (mu20 + mu02)
    ], dtype=float)
