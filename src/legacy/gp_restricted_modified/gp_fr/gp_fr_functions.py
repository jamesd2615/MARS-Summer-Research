"""
GP-FR function set implementing the operators from:
Fan et al., "Genetic Programming for Image Classification: A New Program
Representation With Flexible Feature Reuse", IEEE TEVC 2023.

Layers:
  - Region Detection: RegionR, RegionS
  - Image Filtering: Med, Mean, Min, Max, Gau, GauD, Lap, LoG1, LoG2,
                      Sobel, SobelX, SobelY
  - Feature Extraction: uLBP, SIFT, HOG, Hist, DIF
  - Feature Concatenation: Features_con
  - Classification: Classifiers (SVM, LR, RF, ERF)
"""
import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel, gabor, gaussian
from . import sift_features


# ---------------------------------------------------------------------------
# Restricted GP helpers
# ---------------------------------------------------------------------------

def _as_finite_image(image):
    array = np.asarray(image, dtype=float)

    if array.ndim != 2:
        raise ValueError(
            "Feature primitives expect a two-dimensional image or region."
        )

    if array.size == 0:
        return np.zeros((1, 1), dtype=float)

    return np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _scalar_vector(value):
    scalar = float(
        np.nan_to_num(
            value,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )
    return np.array([scalar], dtype=float)

# ---------------------------------------------------------------------------
# Region Detection Layer (Table I)
# ---------------------------------------------------------------------------

def region_r(image, x, y, width, height):
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 3))
    y = int(np.clip(y, 0, h - 3))
    width = int(np.clip(width, 3, w - x))
    height = int(np.clip(height, 3, h - y))
    return image[y:y + height, x:x + width]


def region_s(image, x, y, length):
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 3))
    y = int(np.clip(y, 0, h - 3))
    max_len = min(w - x, h - y)
    length = int(np.clip(length, 3, max_len))
    return image[y:y + length, x:x + length]


# ---------------------------------------------------------------------------
# Image Filtering Layer (Table II)
# ---------------------------------------------------------------------------

def med_filter(img):
    return ndimage.median_filter(img, size=3)


def mean_filter(img):
    return ndimage.convolve(img, np.full((3, 3), 1.0 / 9.0))


def min_filter(img):
    return ndimage.minimum_filter(img, size=3)


def max_filter(img):
    return ndimage.maximum_filter(img, size=3)


def gau_filter(img, sigma):
    return gaussian(img, sigma=max(1, sigma))


def gau_d_filter(img, sigma, o1, o2):
    return ndimage.gaussian_filter(img, sigma=max(1, sigma), order=[o1, o2])


def lap_filter(img):
    return ndimage.laplace(img)


def log1_filter(img):
    return ndimage.gaussian_laplace(img, sigma=1)


def log2_filter(img):
    return ndimage.gaussian_laplace(img, sigma=2)


def sobel_filter(img):
    return sobel(img)


def sobel_x_filter(img):
    return ndimage.sobel(img, axis=0)


def sobel_y_filter(img):
    return ndimage.sobel(img, axis=1)


# ---------------------------------------------------------------------------
# Feature Extraction Layer (Table III)
# ---------------------------------------------------------------------------

def ulbp_feature(image):
    lbp_img = local_binary_pattern(image, P=8, R=1.5, method='nri_uniform')
    hist, _ = np.histogram(lbp_img, bins=59, range=(0, 59))
    return hist.astype(float)

def rotation_invariant_feature(image):
    """
    Rotation-invariant Local Binary Pattern (LBP) descriptor.
    """

    try:
        image = np.asarray(image, dtype=float)

        if image.ndim != 2 or image.size == 0:
            return np.zeros(10)

        lbp = local_binary_pattern(
            image,
            P=8,
            R=1,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(11),
            range=(0, 10)
        )

        hist = hist.astype(float)

        if hist.sum() > 0:
            hist /= hist.sum()

        return hist

    except Exception:
        return np.zeros(10)

_sift_cache = {}

def sift_feature(image):
    h, w = image.shape[:2]
    min_len = min(h, w)
    if min_len < 4:
        return np.zeros(128)
    try:
        if min_len not in _sift_cache:
            _sift_cache[min_len] = sift_features.SingleSiftExtractor(min_len)
        fea = _sift_cache[min_len].process_image(image[:min_len, :min_len])
        return fea.flatten()
    except Exception:
        return np.zeros(128)


def hog_feature(image):
    try:
        h, w = image.shape[:2]
        ppc = min(8, h, w)
        if ppc < 2:
            return np.zeros(1)
        feat = hog(
            image, orientations=9,
            pixels_per_cell=(ppc, ppc),
            cells_per_block=(1, 1),
            feature_vector=True,
        )
        return feat
    except Exception:
        return np.zeros(1)


def hist_feature(image):
    hist, _ = np.histogram(image.ravel(), bins=64, range=(0.0, 1.0))
    return hist.astype(float)


def dif_feature(image):
    h, w = image.shape[:2]
    if h < 3 or w < 3:
        return np.zeros(8)
    n_regions_h, n_regions_w = min(3, h), min(3, w // 2) if w >= 4 else 1
    rh = h // n_regions_h
    rw = w // n_regions_w
    features = []
    for i in range(n_regions_h):
        for j in range(n_regions_w):
            block = image[i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
            features.append(np.mean(block))
            features.append(np.std(block))
    return np.array(features)


# ---------------------------------------------------------------------------
# Restricted scalar feature primitives
# ---------------------------------------------------------------------------

def mean_feature(image):
    return _scalar_vector(np.mean(_as_finite_image(image)))


def std_feature(image):
    return _scalar_vector(np.std(_as_finite_image(image)))


def min_feature(image):
    return _scalar_vector(np.min(_as_finite_image(image)))


def max_feature(image):
    return _scalar_vector(np.max(_as_finite_image(image)))


def median_feature(image):
    return _scalar_vector(np.median(_as_finite_image(image)))


def energy_feature(image):
    array = _as_finite_image(image)
    return _scalar_vector(np.mean(np.square(array)))


def gradient_x_feature(image):
    array = _as_finite_image(image)
    if array.shape[1] < 2:
        return _scalar_vector(0.0)
    return _scalar_vector(np.mean(np.abs(np.diff(array, axis=1))))


def gradient_y_feature(image):
    array = _as_finite_image(image)
    if array.shape[0] < 2:
        return _scalar_vector(0.0)
    return _scalar_vector(np.mean(np.abs(np.diff(array, axis=0))))


def laplacian_variance_feature(image):
    array = _as_finite_image(image)
    return _scalar_vector(np.var(ndimage.laplace(array)))


def symmetry_feature(image):
    array = _as_finite_image(image)
    half_width = array.shape[1] // 2

    if half_width < 1:
        return _scalar_vector(0.0)

    left_half = array[:, :half_width]
    right_half = np.fliplr(
        array[:, array.shape[1] - half_width:]
    )
    return _scalar_vector(np.mean(np.abs(left_half - right_half)))


def rotation_invariant_entropy_feature(image):
    histogram = np.asarray(
        rotation_invariant_feature(image),
        dtype=float,
    ).reshape(-1)

    histogram = np.nan_to_num(
        histogram,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = float(histogram.sum())
    if total <= 0.0:
        return _scalar_vector(0.0)

    probabilities = histogram / total
    positive = probabilities > 0.0
    entropy = -np.sum(
        probabilities[positive] * np.log2(probabilities[positive])
    )
    return _scalar_vector(entropy)


def rotation_invariant_peak_feature(image):
    histogram = np.asarray(
        rotation_invariant_feature(image),
        dtype=float,
    ).reshape(-1)

    histogram = np.nan_to_num(
        histogram,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = float(histogram.sum())
    if total <= 0.0:
        return _scalar_vector(0.0)

    return _scalar_vector(np.max(histogram / total))


# ---------------------------------------------------------------------------
# Feature Concatenation Layer (Table in Section III-C-4)
# ---------------------------------------------------------------------------

def features_con(*args):
    parts = []

    for argument in args:
        array = np.asarray(argument, dtype=float).reshape(-1)

        if array.size == 0:
            raise ValueError(
                "Cannot concatenate an empty feature vector."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "Cannot concatenate non-finite feature values."
            )

        parts.append(array)

    if not parts:
        raise ValueError(
            "At least one feature vector is required."
        )

    return np.concatenate(parts)


def features_con8(
    feature_1,
    feature_2,
    feature_3,
    feature_4,
    feature_5,
    feature_6,
    feature_7,
    feature_8,
):
    arguments = [
        feature_1,
        feature_2,
        feature_3,
        feature_4,
        feature_5,
        feature_6,
        feature_7,
        feature_8,
    ]

    scalar_parts = []

    for feature_index, argument in enumerate(arguments, start=1):
        array = np.asarray(argument, dtype=float).reshape(-1)

        if array.shape != (1,):
            raise ValueError(
                f"Restricted FC8 child {feature_index} returned "
                f"shape {array.shape}; expected (1,)."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"Restricted FC8 child {feature_index} "
                "returned a non-finite value."
            )

        scalar_parts.append(array)

    result = np.concatenate(scalar_parts)

    if result.shape != (8,):
        raise RuntimeError(
            "Restricted FC8 did not produce exactly eight features."
        )

    return result


# ---------------------------------------------------------------------------
# Classification Layer (Section III-C-5)
# ---------------------------------------------------------------------------

def classifiers(feature_vector, number):
    return feature_vector, int(number)
