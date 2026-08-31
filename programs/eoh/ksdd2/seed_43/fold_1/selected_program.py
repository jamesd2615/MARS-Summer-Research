import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Adaptive ternary thresholds (low/mid/high)
    lo = np.percentile(image, 25)
    hi = np.percentile(image, 75)
    low_bin = (image < lo).mean()
    mid_bin = ((image >= lo) & (image < hi)).mean()
    high_bin = (image >= hi).mean()

    # Radial intensity falloff from center
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    norm_dist = dist / max_dist
    # Weighted mean intensity by ring distance
    radial_energy = np.mean(image * (1 - norm_dist))

    # Edge density (fraction of strong gradient pixels)
    grad = np.hypot(
        np.diff(image, axis=1, prepend=image[:, :1]),
        np.diff(image, axis=0, prepend=image[:1, :])
    )
    grad_thresh = np.mean(grad) + np.std(grad)
    edge_density = (grad > grad_thresh).mean()

    # Low-frequency sinusoidal projection energy (2D cosine basis)
    fy = np.cos(np.pi * np.arange(h) / h).reshape(-1, 1)
    fx = np.cos(np.pi * np.arange(w) / w).reshape(1, -1)
    lowfreq_energy = np.mean(image * fy * fx)

    # Symmetric difference between top-left and bottom-right quadrants (with overlap)
    qh, qw = h // 4, w // 4
    tl = image[:h - qh, :w - qw]
    br = image[qh:, qw:]
    diag_sym = np.mean(np.abs(tl - br))

    # Saturation-like contrast using local range with 3x3 mean pooling
    pooled = np.lib.stride_tricks.sliding_window_view(image, (3, 3)).mean(axis=(2, 3))
    local_range = pooled.max() - pooled.min()

    # Intensity variation ratio (interquartile range / median)
    iqr = np.percentile(image, 75) - np.percentile(image, 25)
    med = np.median(image)
    iqr_ratio = iqr / (med + 1e-10)

    return np.array([
        np.mean(image),
        low_bin / (high_bin + 1e-10),
        mid_bin,
        radial_energy,
        edge_density,
        lowfreq_energy,
        diag_sym,
        iqr_ratio,
    ], dtype=float)
