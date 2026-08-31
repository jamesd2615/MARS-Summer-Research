import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # radial intensity profile: mean intensity on a central disk vs outer ring
    cy, cx = h / 2.0, w / 2.0
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    max_dist = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    inner_mask = dist < 0.4 * max_dist
    outer_mask = dist > 0.6 * max_dist
    inner_mean = np.mean(image[inner_mask]) if np.any(inner_mask) else 0.0
    outer_mean = np.mean(image[outer_mask]) if np.any(outer_mask) else 0.0
    radial_contrast = inner_mean - outer_mean

    # edge orientation ratios: horizontal vs vertical edge magnitudes
    gx = np.abs(np.diff(image, axis=1))
    gy = np.abs(np.diff(image, axis=0))
    gx_mean = np.mean(gx)
    gy_mean = np.mean(gy)
    edge_ratio = gx_mean / (gy_mean + 1e-12)

    # vertical symmetry: compare top half vs bottom half (flipped vertically)
    half_h = h // 2
    top = image[:half_h, :]
    bottom = np.flipud(image[h - half_h:, :])
    vert_sym = np.mean(np.abs(top - bottom))

    # low-intensity proportion using a threshold based on median
    med = np.median(image)
    low_frac = np.mean(image < (med - np.std(image)))

    return np.array([
        inner_mean,
        outer_mean,
        radial_contrast,
        edge_ratio,
        gy_mean - gx_mean,
        vert_sym,
        low_frac,
        np.mean(np.abs(np.diff(image, axis=0)[:, 1:] - np.diff(image, axis=0)[:, :-1])),  # vertical edge curvature proxy
    ], dtype=float)
