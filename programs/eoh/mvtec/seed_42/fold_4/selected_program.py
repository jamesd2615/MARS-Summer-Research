import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Radial intensity profile: mean intensity at different distances from center
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    radii = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_radius = np.sqrt((h/2)**2 + (w/2)**2)
    radial_bins = (radii / max_radius * 4).astype(int)  # 5 bins
    radial_mean = np.zeros(5)
    for i in range(5):
        mask = (radial_bins == i)
        if mask.any():
            radial_mean[i] = np.mean(image[mask])
    radial_spread = np.ptp(radial_mean)  # range of radial means

    # Gradient orientation histogram (bin into 4 quadrants)
    gx = np.gradient(image, axis=1)
    gy = np.gradient(image, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    orient = np.arctan2(gy, gx)
    # Divide into 4 orientation bins (0,45,90,135 degrees approx)
    hist = np.zeros(4)
    for i in range(4):
        angle_start = -np.pi + i * (np.pi/2)
        angle_end = angle_start + np.pi/2
        mask = (orient >= angle_start) & (orient < angle_end)
        hist[i] = np.sum(mag[mask])
    orientation_norm_ratio = hist[0] / (np.sum(hist) + 1e-10)  # dominant orientation proportion

    # Wavelet-like horizontal and vertical detail (Haar approx via differencing)
    h_detail = np.mean(np.abs(image[:, 1:] - image[:, :-1]))
    v_detail = np.mean(np.abs(image[1:, :] - image[:-1, :]))
    detail_ratio = h_detail / (v_detail + 1e-10)

    # Spatial autocorrelation at lag 1 and 2 horizontally
    flat = image.flatten()
    n = len(flat)
    centered = flat - np.mean(flat)
    autocorr1 = np.sum(centered[:-1] * centered[1:]) / (np.sum(centered**2) + 1e-10)
    autocorr2 = np.sum(centered[:-2] * centered[2:]) / (np.sum(centered**2) + 1e-10)
    autocorr_diff = autocorr1 - autocorr2

    # Intensity histogram skewness (using percentiles)
    p25, p50, p75 = np.percentile(image, [25, 50, 75])
    skewness = (p75 - 2*p50 + p25) / (p75 - p25 + 1e-10)

    # Low-frequency ratio (average of 4 corner quadrants vs center)
    center_region = image[int(0.25*h):int(0.75*h), int(0.25*w):int(0.75*w)]
    corner_top_left = image[:int(0.25*h), :int(0.25*w)]
    corner_bottom_right = image[int(0.75*h):, int(0.75*w):]
    low_freq_ratio = np.mean(center_region) / (np.mean(corner_top_left) + np.mean(corner_bottom_right) + 1e-10)

    # Edge density using Laplacian zero-crossings approximation (via second derivative)
    lap = np.zeros_like(image)
    lap[1:-1, 1:-1] = (4*image[1:-1, 1:-1] - image[:-2, 1:-1] - image[2:, 1:-1] - image[1:-1, :-2] - image[1:-1, 2:])
    zero_cross = ((lap[:-1, :-1] * lap[1:, 1:] < 0) | (lap[:-1, 1:] * lap[1:, :-1] < 0)).sum()
    edge_density = zero_cross / ((h-1)*(w-1))

    return np.array([
        radial_spread,
        orientation_norm_ratio,
        detail_ratio,
        autocorr_diff,
        skewness,
        low_freq_ratio,
        edge_density,
        np.median(image)
    ], dtype=float)
