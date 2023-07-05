import cv2
import numpy as np
import matplotlib.pyplot as plt

def autofit_fundus_resolution(fundus, max_size, return_roi=False):
    """
    This function does all the necessary operations to fit a fundus image into a square shape, including automatic removal of black borders 
    It returns the parameters to be given to the reverse_autofit() function to inverse the transformation
    Args:
        fundus (np.ndarray): _description_
        max_size (int): _description_
        return_roi (bool)
    """
    
    image, roi, margins = autocrop_fundus(fundus)
    h, w = image.shape[:-1]
    f = max_size / max((h, w))
    image = cv2.resize(image, fx=f, fy=f, dsize=None)
    h, w = image.shape[:-1]
    padh = (max_size-h)//2, (max_size-h)//2 + ((max_size-h)%2 == 1) 
    padw = (max_size-w)//2, ((max_size-w)%2 == 1) 
    
    reverse_params = {
        'pad':margins,
        'resize':1/f,
        'crop':(padh[0], padh[0]+h, padw[0], padw[0]+w)
    }
    if return_roi:
        roi = cv2.resize(image, fx=f, fy=f, dsize=None)
        return np.pad(image, (padh, padw, (0,0))), np.pad(roi, (padh, padw)), reverse_params
        
    return np.pad(image, (padh, padw, (0,0))), reverse_params
    

def reverse_autofit(image, **forward_params):
    """
    Reverse the transformation done by the autofit function following the parameters given as **kwargs

    Args:
        image (np.ndarray): HxWxC (where C can be > 3, for lesions maps for example)

    Returns:
        np.ndarray: H_new x W_new x C
    """
    crop = forward_params['crop']
    image = image[crop[0]: crop[1], crop[2]:crop[3]]
    
    f = forward_params['resize']
    image = cv2.resize(image, fx=f, fy=f, dsize=None)
    
    pad = forward_params['pad']
    return pad_given_margins(image, pad)
    
def autocrop_fundus(fundus):
    """
    Removes the black border surrounding a fundus image.

    Args:
        fundus (np.ndarray): Images with HxWxC (RGB)

    Returns:
        (np.ndarray, np.ndarray, (int, int, int, int)): Cropped fundus, ROI, Margins (t, b, l, r) 
    """
    h, w, c = fundus.shape
    max_img = np.max(fundus, axis=2)
    max_value = max_img.max()
    threshold = 0.05*max_value
    _, roi = cv2.threshold(max_img, threshold, 1, cv2.THRESH_BINARY)
    not_null_pixels = cv2.findNonZero(roi)
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    
    fundus_croppped = fundus[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    roi_cropped = roi[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    margins_removed = (y_range[0], h-y_range[1], x_range[0], w-x_range[1])
    
    return fundus_croppped, roi_cropped, margins_removed


def pad_given_margins(img, margins):
    """
    

    Args:
        img (np.ndarray): Image
        margins (int, int, int, int): Margins (t,b, l, r)

    Returns:
        np.ndarray : Padded image
    """
    pad_width = ((margins[0], margins[1]), (margins[2], margins[3]), (0,0))
    return np.pad(img, pad_width = pad_width)
    

def open_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def plot_image(image, figsize=(10, 10), axis='off', title=None):
    plt.imshow(image)
    plt.gcf().set_size_inches(figsize)
    plt.axis(axis)
    if title:
        plt.title(title)
    plt.show()