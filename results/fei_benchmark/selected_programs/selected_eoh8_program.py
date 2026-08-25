import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    eps = 1e-12

    half = w // 2
    left = image[:, :half]
    right = np.fliplr(image[:, -half:])
    symmetry = np.mean(np.abs(left - right))

    eye_band = image[int(0.25*h):int(0.45*h), int(0.15*w):int(0.85*w)]
    mouth_band = image[int(0.55*h):int(0.75*h), int(0.20*w):int(0.80*w)]

    dy, dx = np.gradient(image)
    mag = np.sqrt(dx*dx + dy*dy)
    edge_strength = np.mean(mag)
    edge_density = np.mean(mag > np.mean(mag) + np.std(mag))

    rows = np.arange(h).reshape(-1, 1)
    total = np.sum(image) + eps
    vertical_centroid = np.sum(image * rows) / (total * h)

    return np.array([
        np.mean(image),
        np.std(image),
        np.mean(eye_band),
        np.mean(mouth_band),
        symmetry,
        edge_strength,
        edge_density,
        vertical_centroid,
    ], dtype=float)