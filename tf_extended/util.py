import tensorflow as tf
import numpy as np
import cv2

def contour_to_points(contour):
    return np.asarray([c[0] for c in contour])


def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])

def cv2_min_area_rect(xs, ys):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)

    num_rects = xs.shape[0]
    box = np.empty((num_rects, 5))#cx, cy, w, h, theta
    for idx in range(num_rects):
        points = zip(xs[idx, :], ys[idx, :])
        cnt = points_to_contour(points)
        rect = cv2.minAreaRect(cnt)
        cx, cy = rect[0]
        w, h = rect[1]
        theta = rect[2]
        box[idx, :] = [cx, cy, w, h, theta]

    box = np.asarray(box, dtype = xs.dtype)
    return box

def min_area_rect(xs, ys):

    rects = tf.py_func(cv2_min_area_rect, [xs, ys], xs.dtype)
    rects.set_shape([None, 5])
    return rects


def draw_contours(img, contours, idx = -1, color = 1, border_width = 1):

    cv2.drawContours(img, contours, idx, color, border_width)
    return img

def get_shape(img):
    """
    return the height and width of an image
    """
    return np.shape(img)[0:2]

def black(shape):
    if len(np.shape(shape)) >= 2:
        shape = get_shape(shape)
    shape = [int(v) for v in shape]
    return np.zeros(shape, np.uint8)