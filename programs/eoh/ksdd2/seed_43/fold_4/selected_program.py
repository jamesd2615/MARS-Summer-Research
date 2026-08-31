import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Central region for robust local statistics (exclude borders)
    centre = image[
        int(0.15 * h):int(0.85 * h),
        int(0.15 * w):int(0.85 * w)
    ]

    # Gradients (edge magnitudes)
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    grad_mag = np.sqrt(dx[:-1, :]**2 + dy[:, :-1]**2)

    # Quadrant symmetry (top vs bottom, left vs right)
    top = image[:h//2, :]
    bottom = np.flipud(image[h - h//2:, :])
    symmetry_v = np.mean(np.abs(top - bottom))

    # Left vs right symmetry using central column overlap
    left = image[:, :w//2]
    right = np.fliplr(image[:, w - w//2:])
    symmetry_h = np.mean(np.abs(left - right))

    # Proportions of bright and dark pixels relative to median intensity
    med = np.median(image)
    bright_ratio = np.mean(image > med + np.std(image) * 0.5)
    dark_ratio = np.mean(image < med - np.std(image) * 0.5)

    # Regional contrast (difference between central and peripheral means)
    peripheral_top = image[:int(0.1*h), :]
    peripheral_bottom = image[int(0.9*h):, :]
    peripheral_left = image[:, :int(0.1*w)]
    peripheral_right = image[:, int(0.9*w):]
    peripheral_mean = (np.mean(peripheral_top) + np.mean(peripheral_bottom) +
                       np.mean(peripheral_left) + np.mean(peripheral_right)) / 4.0
    regional_contrast = np.mean(centre) - peripheral_mean

    # Second-order moment (variance of gradients as texture)
    texture = np.std(grad_mag)

    return np.array([
        np.mean(centre),
        np.std(centre),
        np.mean(grad_mag),
        np.sqrt(np.mean(dx**2) + np.mean(dy**2)),
        symmetry_v,
        symmetry_h,
        bright_ratio - dark_ratio,
        regional_contrast + texture,
    ], dtype=float)
