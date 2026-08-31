import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Ensure even dimensions for pyramid
    h_pad = h - (h % 2)
    w_pad = w - (w % 2)
    img = image[:h_pad, :w_pad]

    # 1. Pyramid decomposition: downsample by averaging 2x2 blocks
    block_shape = (h_pad // 2, w_pad // 2)
    downsampled = img.reshape(block_shape[0], 2, block_shape[1], 2).mean(axis=(1, 3))

    # 2. Quadrant energy ratios (top/bottom, left/right on downsampled)
    qh, qw = block_shape
    top = downsampled[:qh//2, :]
    bottom = downsampled[qh//2:, :]
    left_q = downsampled[:, :qw//2]
    right_q = downsampled[:, qw//2:]

    top_energy = np.sum(top**2)
    bottom_energy = np.sum(bottom**2) + 1e-9
    left_energy = np.sum(left_q**2)
    right_energy = np.sum(right_q**2) + 1e-9

    # 3. Gradient orientation statistics (0, 45, 90, 135 degrees)
    gy, gx = np.gradient(img)
    magnitude = np.sqrt(gx**2 + gy**2) + 1e-9
    angle = np.arctan2(gy, gx)
    
    # Binned orientation energy at 4 directions
    bins = np.zeros(4)
    for i in range(4):
        mask = (angle >= -np.pi/4 + i*np.pi/2) & (angle < np.pi/4 + i*np.pi/2)
        bins[i] = np.sum(magnitude[mask])
    orientation_entropy = -np.sum((bins / (np.sum(bins)+1e-9)) * np.log(bins / (np.sum(bins)+1e-9) + 1e-9))

    # 4. Intensity range compression ratio (percentile-based)
    p10, p90 = np.percentile(img, [10, 90])
    range_compression = (p90 - p10) / (np.max(img) - np.min(img) + 1e-9)

    # 5. Spatial autocorrelation at lag 1 (row-wise) and lag 1 (col-wise)
    row_auto = np.mean((img[:, :-1] - np.mean(img)) * (img[:, 1:] - np.mean(img))) / (np.var(img) + 1e-9)
    col_auto = np.mean((img[:-1, :] - np.mean(img)) * (img[1:, :] - np.mean(img))) / (np.var(img) + 1e-9)

    # 6. Edge density (high gradient proportion)
    edge_thresh = np.mean(magnitude) + np.std(magnitude)
    edge_density = np.mean(magnitude > edge_thresh)

    # 7. Global skewness (standardized)
    skew = np.mean((img - np.mean(img))**3) / (np.std(img)**3 + 1e-9)

    # 8. Downsampled coefficient of variation (local texture)
    coeff_var = np.std(downsampled) / (np.mean(downsampled) + 1e-9)

    return np.array([
        top_energy / (bottom_energy + top_energy),
        left_energy / (right_energy + left_energy),
        orientation_entropy,
        range_compression,
        row_auto,
        col_auto,
        edge_density,
        skew * coeff_var
    ], dtype=float)
