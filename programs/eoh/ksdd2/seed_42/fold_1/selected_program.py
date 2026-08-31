import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Edge magnitude
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    # Pad image for convolution
    padded = np.pad(image, 1, mode='edge')
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)
    
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mag /= (np.max(edge_mag) + 1e-10)
    
    # Percentile-based regions
    p10 = np.percentile(image, 10)
    p90 = np.percentile(image, 90)
    low_region = image < p10
    high_region = image > p90
    
    # Directional ratios in central band
    central_band = image[int(0.3*h):int(0.7*h), :]
    band_half = central_band.shape[1] // 2
    left_band = central_band[:, :band_half]
    right_band = central_band[:, band_half:]
    
    # Edge density in low vs high regions
    edge_low = np.mean(edge_mag[low_region]) if np.any(low_region) else 0.0
    edge_high = np.mean(edge_mag[high_region]) if np.any(high_region) else 0.0
    
    # Vertical vs horizontal gradient energy
    vert_energy = np.mean(grad_y**2)
    horiz_energy = np.mean(grad_x**2)
    
    # Contrast between central horizontal band and outer areas
    outer = np.concatenate([image[:int(0.3*h), :], image[int(0.7*h):, :]])
    center_mean = np.mean(central_band)
    outer_mean = np.mean(outer)
    
    # Edge density ratio
    edge_density = np.mean(edge_mag > 0.5)
    
    return np.array([
        edge_low - edge_high,           # Edge contrast between dark and bright areas
        vert_energy / (horiz_energy + 1e-10),  # Gradient orientation ratio
        center_mean - outer_mean,       # Center-surround intensity difference
        np.mean(edge_mag) * 10,         # Overall edge magnitude
        np.mean(left_band) - np.mean(right_band),  # Horizontal asymmetry
        np.std(edge_mag[edge_mag > 0.3]) if np.any(edge_mag > 0.3) else 0.0,  # Edge variability
        edge_density * 5,               # Edge density
        np.mean(image > (p10 + p90) / 2)  # Mid-high intensity proportion
    ], dtype=float)
