import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    
    # Quadrant intensity ratios (top-left vs bottom-right, top-right vs bottom-left)
    q1 = image[:h//2, :w//2].mean()
    q2 = image[:h//2, w//2:].mean()
    q3 = image[h//2:, :w//2].mean()
    q4 = image[h//2:, w//2:].mean()
    
    # Gradient magnitude histogram skewed towards low vs high frequencies
    gx = np.abs(np.diff(image, axis=1))
    gy = np.abs(np.diff(image, axis=0))
    # Pad to same shape for combined gradient
    gx_pad = np.pad(gx, ((0,0),(0,1)), mode='edge')
    gy_pad = np.pad(gy, ((0,1),(0,0)), mode='edge')
    grad_mag = np.sqrt(gx_pad**2 + gy_pad**2)
    
    # High-frequency proportion (strong edges)
    high_freq_mask = grad_mag > (np.mean(grad_mag) + np.std(grad_mag))
    high_freq_ratio = np.mean(high_freq_mask)
    
    # Directional contrast: horizontal vs vertical edge strength ratio
    horizontal_strength = np.mean(gx)
    vertical_strength = np.mean(gy)
    direction_balance = horizontal_strength / (vertical_strength + 1e-10)
    
    # Coarse-to-fine intensity variation (downsampled std vs full std)
    coarse = image[::2, ::2]
    coarse_std = np.std(coarse)
    fine_std = np.std(image)
    scale_difference = fine_std - coarse_std
    
    # Local contrast distribution (10th vs 90th percentile of local std)
    local_std = np.zeros((h//4, w//4))
    for i in range(h//4):
        for j in range(w//4):
            block = image[i*4:(i+1)*4, j*4:(j+1)*4]
            local_std[i,j] = np.std(block)
    contrast_spread = np.percentile(local_std, 90) - np.percentile(local_std, 10)
    
    # Entropy-like measure using intensity histogram
    hist, _ = np.histogram(image, bins=16, range=(image.min(), image.max()+1e-10))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    
    return np.array([
        (q1 - q4) / (q1 + q4 + 1e-10),  # diagonal contrast
        (q2 - q3) / (q2 + q3 + 1e-10),  # anti-diagonal contrast
        high_freq_ratio,                  # edge density
        direction_balance,                # orientation bias
        scale_difference,                 # texture coarseness
        contrast_spread,                  # local contrast variability
        entropy,                          # intensity complexity
        np.mean(grad_mag) / (np.std(grad_mag) + 1e-10)  # edge consistency
    ], dtype=float)
