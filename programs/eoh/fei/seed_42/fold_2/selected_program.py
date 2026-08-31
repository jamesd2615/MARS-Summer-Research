import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # 1. Global interquartile range (IQR) as contrast
    q75, q25 = np.percentile(image, [75, 25])
    iqr = q75 - q25

    # 2. Coefficient of variation (robust to scale) 
    std = np.std(image)
    mean = np.mean(image)
    cv = std / (mean + 1e-12)

    # 3. Radial intensity asymmetry: mean absolute difference between inner and outer rings
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = radius.max()
    inner_mask = radius < 0.3 * r_max
    outer_mask = radius > 0.7 * r_max
    inner_mean = np.mean(image[inner_mask]) if np.any(inner_mask) else 0.0
    outer_mean = np.mean(image[outer_mask]) if np.any(outer_mask) else 0.0
    radial_asym = np.abs(inner_mean - outer_mean)

    # 4. Edge density: fraction of pixels where local variation (2x2) exceeds threshold
    # Use a simple 2x2 variance proxy via horizontal and vertical differences
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    grad_mag = np.zeros_like(image)
    grad_mag[:, :-1] += np.abs(dx)
    grad_mag[:-1, :] += np.abs(dy)
    grad_mean = np.mean(grad_mag)
    edge_density = np.mean(grad_mag > (grad_mean + std))

    # 5. Low-frequency dominance: ratio of energy in downsampled (by 2) vs original
    downsampled = image[::2, ::2]
    energy_orig = np.mean(image**2)
    energy_down = np.mean(downsampled**2)
    low_freq_ratio = energy_down / (energy_orig + 1e-12)

    # 6. Vertical periodicity: mean absolute difference between image and shifted by 1 row
    vert_diff = np.abs(np.diff(image, axis=0))
    periodicity = np.mean(vert_diff)

    # 7. Central mass concentration: fraction of total intensity within central 50% area
    centre = image[int(0.25*h):int(0.75*h), int(0.25*w):int(0.75*w)]
    total_intensity = np.sum(image)
    centre_intensity = np.sum(centre)
    mass_concentration = centre_intensity / (total_intensity + 1e-12)

    # 8. Kurtosis of intensity distribution (peakedness)
    if std > 1e-12:
        kurtosis = np.mean((image - mean)**4) / (std**4)
    else:
        kurtosis = 0.0

    return np.array([
        iqr,
        cv,
        radial_asym,
        edge_density,
        low_freq_ratio,
        periodicity,
        mass_concentration,
        kurtosis
    ], dtype=float)
