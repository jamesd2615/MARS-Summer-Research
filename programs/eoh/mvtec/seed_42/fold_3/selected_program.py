import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Global intensity
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    # Vertical asymmetry (top vs bottom)
    half_h = h // 2
    top = image[:half_h, :]
    bottom = np.flipud(image[h-half_h:, :])
    vertical_asym = np.mean(np.abs(top - bottom))

    # Gradient magnitude (combined dx and dy)
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    # Pad to align shapes
    dx_pad = np.pad(dx, ((0,0), (0,1)), mode='edge')
    dy_pad = np.pad(dy, ((0,1), (0,0)), mode='edge')
    grad_mag = np.sqrt(dx_pad**2 + dy_pad**2)
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)

    # Sobel-like edge energy (sum of absolute directional gradients)
    sobely = np.abs(dy).sum() / (h * w)
    sobelx = np.abs(dx).sum() / (h * w)

    # Spatial frequency (mean of absolute second difference)
    d2x = np.diff(image, n=2, axis=1)
    spatial_freq = np.mean(np.abs(d2x))

    # Threshold proportion (using median + std as alternative)
    threshold = np.median(image) + std_intensity
    proportion = np.mean(image > threshold)

    return np.array([
        mean_intensity,
        std_intensity,
        vertical_asym,
        grad_mean,
        grad_std,
        sobely,
        spatial_freq,
        proportion
    ], dtype=float)
