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
# Feature Concatenation Layer (Table in Section III-C-4)
# ---------------------------------------------------------------------------

def features_con(*args):
    parts = []
    for a in args:
        arr = np.asarray(a, dtype=float)
        parts.append(arr.ravel())
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Classification Layer (Section III-C-5)
# ---------------------------------------------------------------------------

def classifiers(feature_vector, number):
    return feature_vector, int(number)
