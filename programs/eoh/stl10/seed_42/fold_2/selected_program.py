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

    # 1. Red-blue mean ratio (log)
    f1 = np.log(np.mean(red) / (np.mean(blue) + 1e-6))

    # 2. Green intensity ratio to total
    f2 = np.mean(green) / (np.mean(red) + np.mean(green) + np.mean(blue) + 1e-6)

    # 3. Central vs peripheral intensity difference (60% center)
    ch, cw = int(0.3*h), int(0.3*w)
    ch_start, cw_start = (h - ch)//2, (w - cw)//2
    centre = intensity[ch_start:ch_start+ch, cw_start:cw_start+cw]
    f3 = np.mean(centre) - np.mean(intensity)

    # 4. Horizontal gradient relative to intensity
    grad_x = np.abs(np.diff(intensity, axis=1))
    f4 = np.mean(grad_x) / (np.mean(intensity) + 1e-6)

    # 5. Vertical gradient relative to intensity
    grad_y = np.abs(np.diff(intensity, axis=0))
    f5 = np.mean(grad_y) / (np.mean(intensity) + 1e-6)

    # 6. Color range standard deviation
    colour_range = np.max(image, axis=2) - np.min(image, axis=2)
    f6 = np.std(colour_range)

    # 7. Red-green channel correlation
    f7 = np.corrcoef(red.ravel(), green.ravel())[0, 1]

    # 8. Top-bottom asymmetry in blue channel
    top = blue[:h//2, :]
    bottom = blue[h//2:, :]
    f8 = np.mean(top) - np.mean(bottom)

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=float)
