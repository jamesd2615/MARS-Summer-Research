import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # 1. Central region mean (already in provided code, keep for baseline)
    centre = image[
        int(0.20 * h):int(0.85 * h),
        int(0.20 * w):int(0.80 * w)
    ]
    centre_mean = np.mean(centre)

    # 2. Block-wise intensity extrema difference (max of block maxima - min of block minima)
    bh, bw = h // 4, w // 4
    blocks = [
        image[y:y+bh, x:x+bw]
        for y in range(0, h - bh + 1, bh)
        for x in range(0, w - bw + 1, bw)
    ]
    block_max = np.max([np.max(b) for b in blocks])
    block_min = np.min([np.min(b) for b in blocks])
    block_extrema_diff = block_max - block_min

    # 3. Spatial frequency ratio (high-frequency energy / low-frequency energy) via simple finite differences
    # High-frequency: second difference magnitude
    d2x = np.diff(image, n=2, axis=1)
    d2y = np.diff(image, n=2, axis=0)
    high_freq = np.mean(d2x**2) + np.mean(d2y**2)
    # Low-frequency: mean of 2x2 pooled (like downsampled) minus original mean
    pooled = (image[0::2, 0::2] + image[0::2, 1::2] + image[1::2, 0::2] + image[1::2, 1::2]) / 4.0
    low_freq = np.mean((pooled - np.mean(image))**2)
    freq_ratio = high_freq / (low_freq + 1e-10)

    # 4. Directional derivative skewness (horizontal minus vertical)
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    skew_x = np.mean((dx - np.mean(dx))**3) / (np.std(dx)**3 + 1e-10)
    skew_y = np.mean((dy - np.mean(dy))**3) / (np.std(dy)**3 + 1e-10)
    derivative_skew = skew_x - skew_y

    # 5. Contrast-normalized intensity deviation (MAD / mean)
    mad = np.mean(np.abs(image - np.mean(image)))
    contrast_dev = mad / (np.mean(image) + 1e-10)

    # 6. Quadrant median spread (range of medians across four quadrants)
    q1 = image[:h//2, :w//2]
    q2 = image[:h//2, w//2:]
    q3 = image[h//2:, :w//2]
    q4 = image[h//2:, w//2:]
    medians = np.array([np.median(q1), np.median(q2), np.median(q3), np.median(q4)])
    quadrant_median_spread = np.max(medians) - np.min(medians)

    # 7. Horizontal symmetry at multiple offsets (mean absolute difference at lag 1 and 2)
    sym1 = np.mean(np.abs(image[:, :-1] - image[:, 1:]))
    sym2 = np.mean(np.abs(image[:, :-2] - image[:, 2:]))
    sym_offset = (sym1 + sym2) / 2.0

    # 8. Proportion of pixels above Otsu-like threshold (mean + std, but using local mean of centre)
    threshold = centre_mean + np.std(image)
    high_prop = np.mean(image > threshold)

    return np.array([
        centre_mean,
        block_extrema_diff,
        freq_ratio,
        derivative_skew,
        contrast_dev,
        quadrant_median_spread,
        sym_offset,
        high_prop
    ], dtype=float)
