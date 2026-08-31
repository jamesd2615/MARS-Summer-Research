import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # 1. Local contrast in overlapping 3x3 patches (mean of local std)
    pad = np.pad(image, 1, mode='edge')
    local_std = np.zeros_like(image)
    for i in range(3):
        for j in range(3):
            patch = pad[i:i+h, j:j+w]
            local_std += (patch - image) ** 2
    local_std = np.sqrt(local_std / 9.0)
    local_contrast = np.mean(local_std)

    # 2. Row-wise intensity oscillation (mean absolute difference between adjacent row means)
    row_means = np.mean(image, axis=1)
    row_oscillation = np.mean(np.abs(np.diff(row_means)))

    # 3. Diagonal gradient energy (mean of abs of diagonal differences)
    diag1 = np.abs(np.diff(image, axis=1)[:-1, :] - np.diff(image, axis=0)[:, 1:])
    diag_energy = np.mean(diag1)

    # 4. Moment skewness (standardized third central moment)
    mean = np.mean(image)
    std = np.std(image)
    skew = np.mean((image - mean) ** 3) / (std ** 3 + 1e-10)

    # 5. Percentile range (90th - 10th percentile)
    p10, p90 = np.percentile(image, [10, 90])
    percentile_range = p90 - p10

    # 6. Binary structure density (fraction of pixels above local median in 5x5 blocks)
    from numpy.lib.stride_tricks import sliding_window_view
    if h >= 5 and w >= 5:
        windows = sliding_window_view(image, (5, 5))
        local_medians = np.median(windows, axis=(2, 3))
        # Upsample medians to original size via edge padding
        med_full = np.pad(local_medians, ((2, 2), (2, 2)), mode='edge')
        binary_density = np.mean(image > med_full)
    else:
        binary_density = 0.0

    # 7. Triangular mask intensity (upper-left triangle vs lower-right triangle)
    tri_mask = np.triu(np.ones((h, w)), k=0).astype(bool)
    upper_left = np.mean(image[tri_mask])
    lower_right = np.mean(image[~tri_mask])
    triangular_diff = upper_left - lower_right

    # 8. Column-wise gradient asymmetry (mean of absolute difference of top-bottom gradients)
    col_grad = np.abs(np.diff(image, axis=0))
    half_h = h // 2
    top_grad = col_grad[:half_h, :]
    bot_grad = col_grad[half_h:, :]
    min_len = min(top_grad.shape[0], bot_grad.shape[0])
    grad_asym = np.mean(np.abs(top_grad[:min_len] - np.flipud(bot_grad[:min_len])))

    return np.array([
        local_contrast,
        row_oscillation,
        diag_energy,
        skew,
        percentile_range,
        binary_density,
        triangular_diff,
        grad_asym
    ], dtype=float)
