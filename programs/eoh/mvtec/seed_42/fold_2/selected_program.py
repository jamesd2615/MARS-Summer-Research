import numpy as np

def extract_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Central region (as before but more focused)
    centre = image[int(0.25*h):int(0.75*h), int(0.25*w):int(0.75*w)]

    # Gradients
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    grad_mag = np.sqrt(dx[:-1, :]**2 + dy[:, :-1]**2)

    # Horizontal and vertical gradient magnitudes separately
    grad_x = np.abs(dx)
    grad_y = np.abs(dy)

    # Edge density (rather than mean abs gradient)
    edge_thresh = np.mean(grad_mag) + np.std(grad_mag)
    edge_density = np.mean(grad_mag > edge_thresh)

    # Vertical symmetry (left-right)
    half_w = w // 2
    left = image[:, :half_w]
    right = np.fliplr(image[:, w-half_w:])
    # Ensure same shape if w odd
    if left.shape[1] != right.shape[1]:
        right = right[:, :left.shape[1]]
    vert_sym = np.mean(np.abs(left - right))

    # Horizontal symmetry (top-bottom)
    half_h = h // 2
    top = image[:half_h, :]
    bottom = np.flipud(image[h-half_h:, :])
    if top.shape[0] != bottom.shape[0]:
        bottom = bottom[:top.shape[0], :]
    hor_sym = np.mean(np.abs(top - bottom))

    # Contrast between center and periphery
    periphery_mean = (np.mean(image) * image.size - np.mean(centre) * centre.size) / (image.size - centre.size)
    centre_periphery_diff = np.mean(centre) - periphery_mean

    # Regional intensity variance ratio (centre vs overall)
    var_ratio = np.std(centre) / (np.std(image) + 1e-9)

    # Proportion of bright pixels in centre vs overall
    overall_bright = np.mean(image > (np.mean(image) + np.std(image)))
    centre_bright = np.mean(centre > (np.mean(centre) + np.std(centre)))
    bright_ratio = centre_bright / (overall_bright + 1e-9)

    return np.array([
        np.mean(grad_x),          # horizontal gradient average
        np.mean(grad_y),          # vertical gradient average
        edge_density,             # edge density
        vert_sym,                 # left-right symmetry
        hor_sym,                  # top-bottom symmetry
        centre_periphery_diff,    # centre vs periphery contrast
        var_ratio,                # centre std / overall std
        bright_ratio,             # centre bright proportion ratio
    ], dtype=float)
