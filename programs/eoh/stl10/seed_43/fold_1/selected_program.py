import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")

    h, w, _ = image.shape

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    intensity = np.mean(image, axis=2)

    # Smaller central crop focusing on core object
    centre = intensity[
        int(0.30 * h):int(0.70 * h),
        int(0.30 * w):int(0.70 * w)
    ]

    # Edge magnitudes from intensity
    dx = np.diff(intensity, axis=1)
    dy = np.diff(intensity, axis=0)
    edge_mag = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)

    # Color saturation and red-green contrast
    sat = np.max(image, axis=2) - np.min(image, axis=2)
    rg_contrast = np.mean(red) - np.mean(green)

    # Normalized vertical gradient energy (relative to horizontal)
    grad_energy_ratio = np.mean(dy ** 2) / (np.mean(dx ** 2) + 1e-8)

    return np.array(
        [
            np.mean(centre),                      # central intensity
            np.std(intensity) / (np.mean(intensity) + 1e-8),  # normalized spread
            np.mean(np.abs(dx)),                  # horizontal gradient
            np.mean(np.abs(dy)),                  # vertical gradient
            np.std(edge_mag),                     # texture variance
            np.mean(sat) - np.std(sat),           # colorfulness measure
            rg_contrast,                          # red-green balance
            grad_energy_ratio,                    # directional gradient ratio
        ],
        dtype=float,
    )
