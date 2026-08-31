import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Feature 1: inner vs outer radial intensity difference
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    inner = image[r < 0.4 * r_max]
    outer = image[r >= 0.7 * r_max]
    f1 = np.mean(inner) - np.mean(outer)

    # Feature 2: standard deviation of radial intensities
    bins = np.linspace(0, r_max, 5)
    radial_means = [np.mean(image[(r >= bins[i]) & (r < bins[i+1])]) for i in range(4)]
    f2 = np.std(radial_means)

    # Feature 3-6: directional edge densities (Sobel-like via central differences)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]

    magnitude = np.sqrt(gx**2 + gy**2)
    eps = 1e-10
    angle = np.arctan2(gy, gx + eps)

    # Four directional masks (0°, 45°, 90°, 135°)
    f3 = np.mean(magnitude[(angle > -np.pi/8) & (angle <= np.pi/8)])
    f4 = np.mean(magnitude[(angle > np.pi/8) & (angle <= 3*np.pi/8)])
    f5 = np.mean(magnitude[(angle > 3*np.pi/8) | (angle <= -3*np.pi/8)])
    f6 = np.mean(magnitude[(angle > -3*np.pi/8) & (angle <= -np.pi/8)])

    # Feature 7: second-order gradient magnitude (Laplacian-like)
    dxx = np.zeros_like(image)
    dyy = np.zeros_like(image)
    dxx[:, 1:-1] = image[:, 2:] - 2*image[:, 1:-1] + image[:, :-2]
    dyy[1:-1, :] = image[2:, :] - 2*image[1:-1, :] + image[:-2, :]
    f7 = np.mean(np.abs(dxx + dyy))

    # Feature 8: vertical symmetry of gradient orientation distribution
    top_half = image[:h//2, :]
    bottom_half = np.flipud(image[h - h//2:, :])
    gx_top = np.diff(top_half, axis=1)
    gx_bottom = np.diff(bottom_half, axis=1)
    # Compare signs of horizontal gradients
    sign_diff = np.mean(np.sign(gx_top) != np.sign(gx_bottom))
    f8 = sign_diff

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=float)
