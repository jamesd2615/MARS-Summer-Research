import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Downsample to a coarse grid using block means
    coarse_h, coarse_w = max(1, h // 4), max(1, w // 4)
    coarse = image[:coarse_h * 4, :coarse_w * 4].reshape(coarse_h, 4, coarse_w, 4).mean(axis=(1, 3))

    # Second-order gradients (Laplacian-like via center differences)
    d2x = np.diff(image, n=2, axis=1)
    d2y = np.diff(image, n=2, axis=0)
    laplace = np.zeros_like(image)
    if d2x.shape[1] > 0 and d2y.shape[0] > 0:
        laplace[1:-1, 1:-1] = d2y[:, 1:-1] + d2x[1:-1, :]
    else:
        laplace = np.abs(d2y).mean() * np.ones_like(image)  # fallback

    # Diagonal symmetry (upper-left vs lower-right triangular regions)
    tri1 = np.triu(image)
    tri2 = np.tril(image)
    diag_sym = np.mean(np.abs(tri1 - tri2.T))

    # Contrast via 10th-90th percentile difference in center
    p10, p90 = np.percentile(coarse, [10, 90])

    # Texture: mean absolute difference from local 3x3 mean
    pad = np.pad(image, 1, mode='edge')
    local_mean = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                  pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                  pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]) / 9.0
    texture = np.mean(np.abs(image - local_mean))

    # Proportion of dark pixels (below mean - std)
    dark_ratio = np.mean(image < (np.mean(image) - np.std(image)))

    return np.array([
        np.mean(coarse),
        np.median(coarse),
        p90 - p10,
        diag_sym,
        np.mean(np.abs(d2x)) if d2x.size > 0 else 0.0,
        np.mean(np.abs(d2y)) if d2y.size > 0 else 0.0,
        texture,
        dark_ratio
    ], dtype=float)
