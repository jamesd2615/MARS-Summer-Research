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

    # Different regional sampling: central and border bands
    central = intensity[
        int(0.35 * h):int(0.65 * h),
        int(0.35 * w):int(0.65 * w)
    ]
    border = np.concatenate([
        intensity[:int(0.15*h), :].ravel(),
        intensity[-int(0.15*h):, :].ravel(),
        intensity[int(0.15*h):-int(0.15*h), :int(0.15*w)].ravel(),
        intensity[int(0.15*h):-int(0.15*h), -int(0.15*w):].ravel()
    ])

    # Channel contrasts (red vs green, blue vs green)
    rg_diff = red - green
    bg_diff = blue - green

    # Saturation: max-min per pixel normalized by intensity (avoid div by zero)
    maxc = np.max(image, axis=2)
    minc = np.min(image, axis=2)
    denominator = np.maximum(intensity, 1e-10)
    saturation = (maxc - minc) / denominator

    # Local edge diversity: standard deviation of gradient magnitudes
    dx = np.diff(intensity, axis=1)
    dy = np.diff(intensity, axis=0)
    grad_mag = np.sqrt((dx[:-1, :]**2 + dy[:, :-1]**2) / 2.0)
    edge_diversity = np.std(grad_mag)

    # Features:
    # 1. Red-green contrast mean
    # 2. Blue-green contrast mean
    # 3. Mean saturation
    # 4. Central vs border intensity ratio (offset)
    # 5. Horizontal gradient mean
    # 6. Vertical gradient mean
    # 7. Edge magnitude standard deviation (diversity)
    # 8. Overall intensity standard deviation

    return np.array([
        np.mean(rg_diff),
        np.mean(bg_diff),
        np.mean(saturation),
        np.mean(central) - np.mean(border),
        np.mean(np.abs(dx)),
        np.mean(np.abs(dy)),
        edge_diversity,
        np.std(intensity)
    ], dtype=float)
