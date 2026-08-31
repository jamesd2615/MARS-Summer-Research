import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Multi-scale regional differences
    quarters = [
        image[:h//2, :w//2],
        image[:h//2, w//2:],
        image[h//2:, :w//2],
        image[h//2:, w//2:]
    ]
    regional_means = [np.mean(q) for q in quarters]
    regional_diff = np.std(regional_means)

    # Directional gradient anisotropies
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    dx_energy = np.mean(dx**2)
    dy_energy = np.mean(dy**2)
    anisotropy = (dy_energy - dx_energy) / (dy_energy + dx_energy + 1e-10)

    # Skewness-based contrast
    centered = image - np.mean(image)
    skewness = np.mean(centered**3) / (np.std(image)**3 + 1e-10)

    # Row/column variability texture
    row_variability = np.mean(np.std(image, axis=1))
    col_variability = np.mean(np.std(image, axis=0))

    # Directional gradient ratio
    grad_ratio = (np.mean(np.abs(dy)) + 1e-10) / (np.mean(np.abs(dx)) + 1e-10)

    # Edge density (proportion of strong gradient pixels)
    grad_mag = np.sqrt(dx[:-1,:]**2 + dy[:,:-1]**2)
    edge_threshold = np.mean(grad_mag) + np.std(grad_mag)
    edge_density = np.mean(grad_mag > edge_threshold)

    return np.array([
        regional_diff,
        anisotropy,
        skewness,
        row_variability,
        col_variability,
        grad_ratio,
        edge_density,
        np.mean(image) * np.std(centre := image[int(0.2*h):int(0.85*h), int(0.2*w):int(0.8*w)]),
    ], dtype=float)
