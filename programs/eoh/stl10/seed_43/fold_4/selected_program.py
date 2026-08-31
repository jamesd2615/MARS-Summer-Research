import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")

    h, w, _ = image.shape

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    intensity = np.mean(image, axis=2)

    # Define central region (20% to 80% of width, 25% to 75% of height for asymmetry)
    top = int(0.25 * h)
    bottom = int(0.75 * h)
    left = int(0.20 * w)
    right = int(0.80 * w)
    centre = intensity[top:bottom, left:right]

    # Horizontal and vertical gradient magnitudes
    dx = np.diff(intensity, axis=1)
    dy = np.diff(intensity, axis=0)

    # Colour spread (max-min across RGB)
    colour_range = np.max(image, axis=2) - np.min(image, axis=2)

    # Colour contrast: red vs blue ratio
    red_blue_contrast = np.mean(red) - np.mean(blue)

    # Centre vs outer intensity difference (outer = total minus centre)
    centre_mean = np.mean(centre)
    total_mean = np.mean(intensity)
    # Avoid double counting centre in outer region
    centre_area = centre.size
    total_area = h * w
    outer_mean = (total_mean * total_area - centre_mean * centre_area) / (total_area - centre_area)
    centre_outer_diff = centre_mean - outer_mean

    # Vertical symmetry: mean absolute difference between top and bottom halves
    half_h = h // 2
    top_half = intensity[:half_h, :]
    bottom_half = intensity[h - half_h:, :]  # take same height for comparison
    if top_half.shape[0] != bottom_half.shape[0]:
        bottom_half = bottom_half[:top_half.shape[0], :]
    symmetry_diff = np.mean(np.abs(top_half - bottom_half))

    # Colour saturation (mean of colour_range normalized by intensity, avoid division by zero)
    eps = 1e-10
    saturation = np.mean(colour_range / (intensity + eps))

    return np.array(
        [
            np.mean(red),
            np.mean(green),
            np.mean(blue),
            np.std(intensity),
            centre_outer_diff,
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            saturation,
        ],
        dtype=float,
    )
