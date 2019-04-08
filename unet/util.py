import numpy as np
import cv2

def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])


def get_neighbours(x, y):
   return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
            (x - 1, y),                 (x + 1, y), \
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h

def find_contours(mask, method = None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP,
                                       method = method)
    except:
        _, contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP,
                                          method = method)
    return contours

def draw_contours(img, contours, idx = -1, color = 1, border_width = 1):

    cv2.drawContours(img, contours, idx, color, border_width)
    return img