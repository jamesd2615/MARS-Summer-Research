import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    
    # Multiscale box mean energies (coarse to fine)
    def pooled_energy(img, block_h, block_w):
        ph, pw = h // block_h, w // block_w
        pooled = img[:ph*block_h, :pw*block_w].reshape(ph, block_h, pw, block_w).mean(axis=(1,3))
        return np.mean(pooled**2)
    
    energy_coarse = pooled_energy(image, 16, 16)
    energy_fine = pooled_energy(image, 4, 4)
    energy_ratio = energy_fine / (energy_coarse + 1e-8)
    
    # Vertical gradient skewness (directional asymmetry)
    dy = np.diff(image, axis=0)
    dy_flat = dy.flatten()
    dy_mean = np.mean(dy_flat)
    dy_std = np.std(dy_flat)
    skew = np.mean(((dy_flat - dy_mean) / (dy_std + 1e-8))**3)
    
    # Radial intensity dispersion (distance-weighted)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h/2.0, w/2.0
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    dist_norm = dist / dist.max()
    weighted_std = np.sqrt(np.mean(dist_norm * (image - np.mean(image))**2))
    
    # Quadrant contrast (differences between four quadrants)
    qh, qw = h // 2, w // 2
    q1 = image[:qh, :qw]
    q2 = image[:qh, qw:]
    q3 = image[qh:, :qw]
    q4 = image[qh:, qw:]
    quad_contrast = np.std([np.mean(q1), np.mean(q2), np.mean(q3), np.mean(q4)])
    
    # Horizontal profile deviation (mean absolute difference from row-wise means)
    row_means = np.mean(image, axis=1, keepdims=True)
    row_dev = np.mean(np.abs(image - row_means))
    
    # Top-bottom intensity asymmetry
    top = image[:qh, :]
    bottom = image[qh:, :]
    tb_asym = np.mean(np.abs(top - np.flipud(bottom)))
    
    # Central dark proportion (relative to overall mean)
    central = image[int(0.3*h):int(0.7*h), int(0.3*w):int(0.7*w)]
    dark_prop = np.mean(central < np.mean(image))
    
    return np.array([energy_coarse, energy_fine, energy_ratio, skew, 
                     weighted_std, quad_contrast, row_dev, tb_asym, dark_prop], dtype=float)[:8]
