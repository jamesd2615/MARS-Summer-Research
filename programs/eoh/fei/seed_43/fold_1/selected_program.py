import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Horizontal and vertical Sobel-like gradients (using central differences)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Vertical symmetry (top vs bottom)
    half_h = h // 2
    top = image[:half_h, :]
    bottom = np.flipud(image[h - half_h:, :])

    # Local contrast: difference between high and low percentile intensities
    p_low = np.percentile(image, 25)
    p_high = np.percentile(image, 75)
    local_contrast = p_high - p_low

    return np.array(
        [
            np.mean(image),
            np.std(image),
            np.mean(grad_mag),
            np.std(grad_mag),
            np.mean(np.abs(gx)),
            np.mean(np.abs(gy)),
            np.mean(np.abs(top - bottom)),
            local_contrast,
        ],
        dtype=float,
    )
