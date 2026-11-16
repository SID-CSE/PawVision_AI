"""Alternative image representations: grayscale, HOG."""
import numpy as np


def rgb_to_gray(x_uint8):
    """(N,H,W,3) uint8 -> (N,H,W) float32 in [0,1] using ITU-R 601 luma weights."""
    w = np.array([0.299, 0.587, 0.114], dtype="float32")
    return (x_uint8.astype("float32") @ w) / 255.0


def hog_features(x_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Histogram of Oriented Gradients for each (H,W) grayscale image in [0,1]."""
    from skimage.feature import hog
    feats = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, block_norm="L2-Hys", feature_vector=True)
             for img in x_gray]
    return np.asarray(feats, dtype="float32")
