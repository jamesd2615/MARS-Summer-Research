import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Different central crop: tighter and slightly offset
    centre = image[
        int(0.30 * h):int(0.75 * h),
        int(0.25 * w):int(0.70 * w)
    ]

    # Sobel-like gradients (using central differences with padding)
    padded = np.pad(image, 1, mode='edge')
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Vertical symmetry on a resized core (using downsampling via slicing)
    step = max(1, h // 32)
    small = image[::step, ::step]
    sh, sw = small.shape
    half = sw // 2
    left = small[:, :half]
    right = np.fliplr(small[:, sw-half:])

    # Local contrast via percentile range
    p_low, p_high = np.percentile(image, [15, 85])

    # Skewness (third central moment normalized)
    mean = np.mean(image)
    std = np.std(image)
    skew = np.mean((image - mean)**3) / (std**3 + 1e-12)

    return np.array(
        [
            np.median(centre),              # robust central intensity
            np.std(centre),                 # central variability
            np.mean(grad_mag),              # overall gradient magnitude
            np.std(grad_mag),               # gradient variability
            np.mean(np.abs(left - right)),  # vertical symmetry difference
            p_high - p_low,                 # local contrast range
            np.mean(image > mean + 0.5 * std),  # high-intensity proportion (different threshold)
            skew,                           # intensity skewness
        ],
        dtype=float,
    )
