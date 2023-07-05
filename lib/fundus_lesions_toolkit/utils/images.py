import cv2
import numpy as np


def fit_fundus_to_shape(fundus, shape):
    pass



def autocrop_fundus(fundus):
    """
    Removes the black border surrounding a fundus image.

    Args:
        fundus (np.ndarray): Images with HxWxC (RGB)

    Returns:
        (np.ndarray, np.ndarray, (int, int, int, int)): Cropped fundus, ROI, Margins (t, b, l, r) 
    """
    max_img = np.max(fundus, axis=2)
    max_value = max_img.max()
    threshold = 0.05*max_value
    
    _, roi = cv2.threshold(fundus, threshold, 1, cv2.THRESH_BINARY)
    not_null_pixels = cv2.findNonZero(roi)
    
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    
    fundus_croppped = fundus[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    roi_cropped = roi[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    margins_removed = (y_range[0], y_range[1], x_range[0], x_range[1])
    return fundus_croppped, roi_cropped, margins_removed


def reverse_autocrop(fundus_cropped, margins):
    pass