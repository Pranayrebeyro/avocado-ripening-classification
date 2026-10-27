"""Extract color and texture features from images."""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMAGE_SIZE = (128, 128)
LBP_POINTS = 16
LBP_RADIUS = 2
LBP_BINS = LBP_POINTS + 3


def extract_image_features(image_path):
    """
    Extract a 123-dimensional feature vector.

    Features:
        HSV histograms: 56 values
        LAB histograms: 48 values
        LBP histogram: 19 values
    """
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # HSV: 24 + 16 + 16 = 56 features.
    hsv_bins = [
        (0, 24, [0, 180]),
        (1, 16, [0, 256]),
        (2, 16, [0, 256]),
    ]

    for channel, bins, value_range in hsv_bins:
        histogram = cv2.calcHist(
            [hsv], [channel], None, [bins], value_range
        )
        histogram = cv2.normalize(histogram, histogram).flatten()
        features.extend(histogram)

    # LAB: 16 bins for each of three channels = 48 features.
    for channel in range(3):
        histogram = cv2.calcHist(
            [lab], [channel], None, [16], [0, 256]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()
        features.extend(histogram)

    # Uniform LBP: 19 histogram bins.
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )
    lbp_histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_BINS + 1),
        range=(0, LBP_BINS),
    )
    lbp_histogram = lbp_histogram.astype(np.float32)
    lbp_histogram /= max(lbp_histogram.sum(), 1)
    features.extend(lbp_histogram)

    feature_vector = np.asarray(features, dtype=np.float32)

    if feature_vector.shape != (123,):
        raise RuntimeError(
            f"Expected 123 features, got {feature_vector.shape}"
        )

    if not np.isfinite(feature_vector).all():
        raise ValueError(f"Non-finite features extracted from {image_path}")

    return feature_vector
