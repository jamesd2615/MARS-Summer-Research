import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    
    # Intensity moments (skewness and kurtosis via standardized moments)
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        skew = np.mean((image - mean)**3) / (std**3)
        kurt = np.mean((image - mean)**4) / (std**4) - 3
    else:
        skew = 0.0
        kurt = 0.0
    
    # Center-to-edge contrast (difference between center and border means)
    center_region = image[int(0.3*h):int(0.7*h), int(0.3*w):int(0.7*w)]
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)] = False
    border_mean = np.mean(image[border_mask]) if np.any(border_mask) else mean
    center_edge_contrast = np.mean(center_region) - border_mean
    
    # Gradient directional ratio (horizontal vs vertical activity)
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    grad_x = np.mean(np.abs(dx))
    grad_y = np.mean(np.abs(dy))
    grad_ratio = grad_x / (grad_y + 1e-10)
    
    # Texture coarseness via local range (max-min in small blocks)
    block_h, block_w = 4, 4
    h_blocks = h // block_h
    w_blocks = w // block_w
    if h_blocks > 0 and w_blocks > 0:
        blocks = image[:h_blocks*block_h, :w_blocks*block_w].reshape(
            h_blocks, block_h, w_blocks, block_w
        )
        block_max = blocks.max(axis=(1, 3))
        block_min = blocks.min(axis=(1, 3))
        local_range = np.mean(block_max - block_min)
    else:
        local_range = np.ptp(image)
    
    # Vertical symmetry (correlation between top and bottom halves)
    half_h = h // 2
    top = image[:half_h, :]
    bottom = np.flipud(image[h-half_h:, :])
    if half_h > 0 and np.std(top) > 0 and np.std(bottom) > 0:
        vert_sym = np.mean((top - np.mean(top)) * (bottom - np.mean(bottom))) / (
            np.std(top) * np.std(bottom)
        )
    else:
        vert_sym = 0.0
    
    # High-intensity spatial dispersion (std of coordinates above threshold)
    threshold = mean + std
    bright_mask = image > threshold
    if np.any(bright_mask):
        ys, xs = np.nonzero(bright_mask)
        spatial_dispersion = np.std(xs) / w + np.std(ys) / h
    else:
        spatial_dispersion = 0.0
    
    return np.array([
        skew,
        kurt,
        center_edge_contrast,
        grad_ratio,
        local_range,
        vert_sym,
        spatial_dispersion,
        np.max(image) - np.min(image)  # overall intensity spread
    ], dtype=float)
