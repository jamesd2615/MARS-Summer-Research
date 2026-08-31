import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    mean_value = np.mean(image)
    std_value = np.std(image)

    dx = np.empty_like(image)
    dy = np.empty_like(image)

    dx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    dx[:, 0] = image[:, 1] - image[:, 0]
    dx[:, -1] = image[:, -1] - image[:, -2]

    dy[1:-1, :] = image[2:, :] - image[:-2, :]
    dy[0, :] = image[1, :] - image[0, :]
    dy[-1, :] = image[-1, :] - image[-2, :]

    grad_magnitude = np.sqrt(dx * dx + dy * dy)
    mean_gradient = np.mean(grad_magnitude)
    gradient_std = np.std(grad_magnitude)

    dark_threshold = mean_value - std_value
    bright_threshold = mean_value + std_value

    dark_proportion = np.mean(image < dark_threshold)
    bright_proportion = np.mean(image > bright_threshold)

    h, w = image.shape
    cells = 4
    block_h, block_w = h // cells, w // cells

    block_means = (
        image[:block_h * cells, :block_w * cells]
        .reshape(cells, block_h, cells, block_w)
        .mean(axis=(1, 3))
    )
    regional_std = np.std(block_means)

    padded = np.pad(image, 1, mode='reflect')
    local_mean = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0
    local_contrast = np.mean(np.abs(image - local_mean))

    return np.array([
        mean_value,
        std_value,
        mean_gradient,
        gradient_std,
        dark_proportion,
        bright_proportion,
        regional_std,
        local_contrast,
    ], dtype=float)