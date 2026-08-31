import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # 1. Normalized entropy (deterministic binning)
    bins = np.linspace(image.min(), image.max() + 1e-12, 16)
    hist, _ = np.histogram(image, bins=bins)
    probs = hist / (h * w)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs)) / np.log(16)

    # 2. Local binary pattern uniformity (using 3x3 neighborhoods)
    padded = np.pad(image, 1, mode='edge')
    center = padded[1:-1, 1:-1]
    neighbors = np.stack([
        padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
        padded[1:-1, 2:], padded[2:, 2:], padded[2:, 1:-1],
        padded[2:, :-2], padded[1:-1, :-2]
    ])
    lbp = (neighbors >= center).astype(np.uint8)
    # Count number of 0/1 transitions around the ring
    transitions = np.sum(np.abs(np.diff(lbp, axis=0, append=lbp[0:1])), axis=0)
    uniformity = np.mean(transitions <= 2)

    # 3. Row periodicity via autocorrelation at half-height
    row_mean = np.mean(image, axis=1)
    row_centered = row_mean - np.mean(row_mean)
    lag = max(1, h // 5)
    if h > 2 * lag:
        corr = np.corrcoef(row_centered[:-lag], row_centered[lag:])[0, 1]
    else:
        corr = 0.0
    row_period = np.nan_to_num(corr)

    # 4. Column periodicity similarly
    col_mean = np.mean(image, axis=0)
    col_centered = col_mean - np.mean(col_mean)
    if w > 2 * lag:
        corr_col = np.corrcoef(col_centered[:-lag], col_centered[lag:])[0, 1]
    else:
        corr_col = 0.0
    col_period = np.nan_to_num(corr_col)

    # 5. Asymmetry ratio: mean of upper half vs lower half
    top = np.mean(image[:h//2])
    bottom = np.mean(image[h//2:])
    asym_ud = (top - bottom) / (top + bottom + 1e-12)

    # 6. Left-right asymmetry
    left = np.mean(image[:, :w//2])
    right = np.mean(image[:, w//2:])
    asym_lr = (left - right) / (left + right + 1e-12)

    # 7. Gradient magnitude skewness (alternative computation)
    gy, gx = np.gradient(image)
    mag = np.hypot(gx, gy)
    mag_std = np.std(mag)
    if mag_std > 0:
        grad_skew = np.mean(((mag - np.mean(mag)) / mag_std) ** 3)
    else:
        grad_skew = 0.0

    # 8. High-frequency content ratio (Laplacian energy normalized)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from numpy.lib.stride_tricks import sliding_window_view
    if h >= 3 and w >= 3:
        windows = sliding_window_view(image, (3, 3))
        lap_vals = np.tensordot(windows, laplacian, axes=([2, 3], [0, 1]))
        high_freq = np.mean(lap_vals ** 2) / (np.var(image) + 1e-12)
    else:
        high_freq = 0.0

    return np.array([
        entropy,
        uniformity,
        row_period,
        col_period,
        asym_ud,
        asym_lr,
        grad_skew,
        high_freq
    ], dtype=float)
