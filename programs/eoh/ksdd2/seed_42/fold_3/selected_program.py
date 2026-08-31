import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # 1. Row-wise linear trend magnitude (average slope)
    x = np.arange(w, dtype=float)
    x_mean = x - np.mean(x)
    row_slopes = np.sum(image * x_mean, axis=1) / np.sum(x_mean**2)
    row_trend = np.mean(np.abs(row_slopes))

    # 2. Column-wise quadratic curvature magnitude
    y = np.arange(h, dtype=float)
    y_mean = y - np.mean(y)
    y2_mean = y_mean**2 - np.mean(y_mean**2)
    denom = np.sum(y_mean**4) - (np.sum(y_mean**2)**2)/h + 1e-10
    col_curv = np.zeros(w)
    for i in range(w):
        coeffs = np.polyfit(y, image[:, i], 2)
        col_curv[i] = coeffs[0]
    quadratic_curv = np.mean(np.abs(col_curv))

    # 3. Circular band intensity variance (polar rings)
    cy, cx = h/2.0, w/2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_max = np.max(r)
    ring_means = []
    for k in range(5):
        mask = (r >= k*r_max/5) & (r < (k+1)*r_max/5)
        if mask.sum() > 0:
            ring_means.append(np.mean(image[mask]))
        else:
            ring_means.append(0.0)
    ring_variance = np.std(ring_means)

    # 4. Gradient orientation entropy (8-bin histogram)
    gy, gx = np.gradient(image)
    angle = np.arctan2(gy, gx)
    angle_flat = angle.ravel()
    hist, _ = np.histogram(angle_flat, bins=8, range=(-np.pi, np.pi))
    prob = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(prob * np.log(prob + 1e-10))

    # 5. Co-occurrence contrast via shifted dot product (horizontal offset 2)
    shift = 2
    slice1 = image[:, 0:w-shift]
    slice2 = image[:, shift:]
    cooccurrence = np.mean((slice1 - slice2)**2)

    # 6. Inter-quadrant mean difference (top vs bottom)
    top = image[:h//2, :]
    bottom = image[h//2:, :]
    top_bottom_diff = np.abs(np.mean(top) - np.mean(bottom))

    # 7. Intensity range within center 50%
    center = image[int(0.25*h):int(0.75*h), int(0.25*w):int(0.75*w)]
    center_range = np.ptp(center)

    # 8. Proportion of pixels with large local contrast (max-min in 3x3 neighborhood)
    padded = np.pad(image, 1, mode='edge')
    patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    local_min = patches.min(axis=(2, 3))
    local_max = patches.max(axis=(2, 3))
    contrast_map = local_max - local_min
    threshold_c = np.mean(contrast_map) + np.std(contrast_map)
    high_contrast_prop = np.mean(contrast_map > threshold_c)

    return np.array([
        row_trend,
        quadratic_curv,
        ring_variance,
        entropy,
        cooccurrence,
        top_bottom_diff,
        center_range,
        high_contrast_prop,
    ], dtype=float)
