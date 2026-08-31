import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Sobel-like edge magnitudes (deterministic)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    if w >= 3:
        gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    if h >= 3:
        gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    edge_mag = np.sqrt(gx**2 + gy**2)

    # Row and column variance (orientation/structure)
    row_var = np.var(image, axis=1)
    col_var = np.var(image, axis=0)
    row_var_mean = np.mean(row_var)
    col_var_mean = np.mean(col_var)
    var_ratio = row_var_mean / (col_var_mean + 1e-12)

    # Local binary pattern like texture: compare each pixel to its right and below neighbour
    right_diff = np.zeros_like(image)
    down_diff = np.zeros_like(image)
    if w > 1 and h > 1:
        right_diff[:, :-1] = (image[:, 1:] > image[:, :-1]).astype(float)
        down_diff[:-1, :] = (image[1:, :] > image[:-1, :]).astype(float)
    texture = np.mean(right_diff * down_diff)  # high when both horizontal and vertical transitions agree

    # Quadrant symmetry (top-bottom and left-right intensity differences)
    top_bottom = np.abs(np.mean(image[:h//2, :]) - np.mean(image[h//2:, :]))
    left_right = np.abs(np.mean(image[:, :w//2]) - np.mean(image[:, w//2:]))

    # Low-intensity proportion (dark pixel ratio)
    low_thresh = np.mean(image) - np.std(image)
    dark_ratio = np.mean(image < low_thresh)

    # Edge magnitude statistics
    edge_mean = np.mean(edge_mag)
    edge_std = np.std(edge_mag)

    return np.array([
        var_ratio,
        top_bottom,
        left_right,
        texture,
        edge_mean,
        edge_std,
        dark_ratio,
        np.mean(edge_mag > np.mean(edge_mag) + np.std(edge_mag))
    ], dtype=float)
