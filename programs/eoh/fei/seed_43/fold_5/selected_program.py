import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    
    # Radial intensity profile: average intensity at different distance bands from center
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
    max_dist = distances.max()
    dist_bands = np.floor(distances / (max_dist / 4.0)).astype(int)
    dist_bands = np.clip(dist_bands, 0, 3)
    radial_means = [np.mean(image[dist_bands == i]) for i in range(4)]
    radial_spread = np.std(radial_means)
    
    # Local binary pattern count: proportion of pixels where right neighbor is brighter
    right_neighbor = np.pad(image, ((0,0),(0,1)), mode='edge')[:, 1:]
    lbp_right = (right_neighbor > image).astype(float)
    lbp_proportion = np.mean(lbp_right)
    
    # Third central moment (skewness-like) of intensity distribution
    mean_intensity = np.mean(image)
    diff = image - mean_intensity
    skewness = np.mean(diff**3) / (np.std(image)**3 + 1e-12)
    
    # Interquartile range
    flat = image.ravel()
    q75, q25 = np.percentile(flat, [75, 25])
    iqr = q75 - q25
    
    # Vertical edge density: count significant vertical edges
    dy = np.abs(np.diff(image, axis=0))
    edge_thresh = np.mean(dy) + np.std(dy)
    vertical_edge_density = np.mean(dy > edge_thresh)
    
    # Horizontal line pattern: mean absolute difference between rows
    row_means = np.mean(image, axis=1)
    row_diff = np.abs(np.diff(row_means))
    row_variability = np.mean(row_diff)
    
    # Moment-based asymmetry: ratio of intensity-weighted x-coordinate vs y-coordinate variance
    total_mass = np.sum(image) + 1e-12
    x_w = np.sum(image * x_coords) / total_mass
    y_w = np.sum(image * y_coords) / total_mass
    x_var = np.sum(image * (x_coords - x_w)**2) / total_mass
    y_var = np.sum(image * (y_coords - y_w)**2) / total_mass
    moment_ratio = x_var / (y_var + 1e-12)
    
    # Fourier-like low-frequency energy (using 2D convolution with averaging kernel)
    kernel = np.ones((3,3)) / 9.0
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(image, 1, mode='edge')
    windows = sliding_window_view(padded, (3,3))
    local_means = np.mean(windows, axis=(2,3))
    low_freq_energy = np.mean(local_means**2) / (np.mean(image**2) + 1e-12)
    
    return np.array([
        radial_spread,
        lbp_proportion,
        skewness,
        iqr,
        vertical_edge_density,
        row_variability,
        moment_ratio,
        low_freq_energy
    ], dtype=float)
