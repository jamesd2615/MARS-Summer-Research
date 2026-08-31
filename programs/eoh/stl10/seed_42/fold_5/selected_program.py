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

    # Central region (different crop than before)
    centre = intensity[
        int(0.20 * h):int(0.85 * h),
        int(0.20 * w):int(0.80 * w)
    ]

    # Horizontal and vertical gradients
    dx = np.diff(intensity, axis=1)
    dy = np.diff(intensity, axis=0)

    # Colour saturation (max-min per pixel)
    colour_range = np.max(image, axis=2) - np.min(image, axis=2)

    # Central vs peripheral luminance contrast (peripheral = outside centre)
    mask = np.zeros_like(intensity, dtype=bool)
    mask[int(0.20 * h):int(0.85 * h), int(0.20 * w):int(0.80 * w)] = True
    peripheral = intensity[~mask]
    centre_peripheral_contrast = np.mean(centre) - np.mean(peripheral)

    # Gradient energy ratio (horizontal / (horizontal + vertical + eps))
    h_energy = np.mean(dx ** 2)
    v_energy = np.mean(dy ** 2)
    gradient_ratio = h_energy / (h_energy + v_energy + 1e-12)

    # Saturation spread (std of colour range)
    saturation_spread = np.std(colour_range)

    return np.array(
        [
            np.median(red),
            np.median(green),
            np.median(blue),
            np.std(intensity),
            centre_peripheral_contrast,
            gradient_ratio,
            np.mean(np.abs(dx)) - np.mean(np.abs(dy)),
            saturation_spread,
        ],
        dtype=float,
    )
