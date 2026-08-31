import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Larger central crop (10% to 90%)
    centre = image[
        int(0.10 * h):int(0.90 * h),
        int(0.10 * w):int(0.90 * w)
    ]

    # Sobel-like gradients using central differences
    dx = (image[:, 2:] - image[:, :-2]) / 2.0
    dy = (image[2:, :] - image[:-2, :]) / 2.0

    # Diagonal symmetry (top-left vs bottom-right)
    tl = image[:h//2, :w//2]
    br = image[h//2:, w//2:]
    # Make same shape by cropping to min size
    min_h = min(tl.shape[0], br.shape[0])
    min_w = min(tl.shape[1], br.shape[1])
    tl_crop = tl[:min_h, :min_w]
    br_crop = br[:min_h, :min_w]

    # Two different thresholds: mean and mean - 0.5*std
    t1 = np.mean(image) + 0.5 * np.std(image)
    t2 = np.mean(image) - 0.5 * np.std(image)

    return np.array(
        [
            np.mean(centre),           # central mean
            np.std(centre),            # central std
            np.mean(np.abs(dx)),       # horizontal gradient
            np.mean(np.abs(dy)),       # vertical gradient
            np.mean(np.abs(tl_crop - br_crop)),  # diagonal symmetry
            np.mean(image > t1),       # high threshold proportion
            np.mean(image < t2),       # low threshold proportion
            np.mean(np.abs(image - np.median(image))),  # median absolute deviation
        ],
        dtype=float,
    )
