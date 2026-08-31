import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # 1. Row-wise mean profile variance (vertical texture)
    row_means = np.mean(image, axis=1)
    feat1 = np.std(row_means)

    # 2. Column-wise mean profile variance (horizontal texture)
    col_means = np.mean(image, axis=0)
    feat2 = np.std(col_means)

    # 3. Laplacian approximation (second derivative) mean absolute
    lap = np.zeros_like(image)
    lap[1:-1, 1:-1] = (image[2:, 1:-1] + image[:-2, 1:-1] +
                       image[1:-1, 2:] + image[1:-1, :-2] -
                       4 * image[1:-1, 1:-1])
    feat3 = np.mean(np.abs(lap))

    # 4. Quadrant energy ratio (top-left vs bottom-right)
    tl = image[:h//2, :w//2]
    br = image[h//2:, w//2:]
    feat4 = np.mean(tl**2) / (np.mean(br**2) + 1e-12)

    # 5. Quadrant energy ratio (top-right vs bottom-left)
    tr = image[:h//2, w//2:]
    bl = image[h//2:, :w//2]
    feat5 = np.mean(tr**2) / (np.mean(bl**2) + 1e-12)

    # 6. Horizontal gradient absolute skewness-like measure (asymmetry in edges)
    dx = np.diff(image, axis=1)
    abs_dx = np.abs(dx)
    mean_abs_dx = np.mean(abs_dx)
    feat6 = np.mean((abs_dx - mean_abs_dx)**3) / (mean_abs_dx**3 + 1e-12)

    # 7. Fraction of pixels above mean + 0.5*std (tail proportion)
    high_thresh = np.mean(image) + 0.5 * np.std(image)
    feat7 = np.mean(image > high_thresh)

    # 8. Coefficient of variation of row-wise standard deviations (spatial uniformity)
    row_stds = np.std(image, axis=1)
    feat8 = np.std(row_stds) / (np.mean(row_stds) + 1e-12)

    return np.array([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8], dtype=float)
