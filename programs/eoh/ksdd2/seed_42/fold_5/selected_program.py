import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Larger central region (15% to 90%)
    centre = image[
        int(0.15 * h):int(0.90 * h),
        int(0.15 * w):int(0.90 * w)
    ]

    # Sobel-like gradients (using central differences with padding)
    padded = np.pad(image, 1, mode='edge')
    dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    edge_mag = np.sqrt(dx**2 + dy**2)

    # Quadrant symmetry: top-left vs bottom-right and top-right vs bottom-left
    qh, qw = h // 2, w // 2
    tl = image[:qh, :qw]
    br = image[qh:2*qh, qw:2*qw]
    tr = image[:qh, qw:2*qw]
    bl = image[qh:2*qh, :qw]
    # Ensure same size by cropping
    min_h = min(tl.shape[0], br.shape[0])
    min_w = min(tl.shape[1], br.shape[1])
    tl = tl[:min_h, :min_w]
    br = br[:min_h, :min_w]
    tr = tr[:min_h, :min_w]
    bl = bl[:min_h, :min_w]
    sym_diag = np.mean(np.abs(tl - br))
    sym_anti = np.mean(np.abs(tr - bl))

    # Lower threshold: mean + 0.5*std
    threshold = np.mean(image) + 0.5 * np.std(image)

    # Coefficient of variation for global intensity
    global_std = np.std(image)
    global_mean = np.mean(image)
    cv = global_std / (global_mean + 1e-9)

    return np.array(
        [
            global_mean,
            cv,
            np.mean(centre),
            np.std(centre),
            np.mean(edge_mag),
            np.std(edge_mag),
            sym_diag - sym_anti,
            np.mean(image > threshold),
        ],
        dtype=float,
    )
