import logging

import cv2
import ml_serving.utils.helpers as helpers
import numpy as np

LOG = logging.getLogger(__name__)
PARAMS = {
    'test': 1
}
treshhold = 0.3
split_counts = []


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)


def box_intersection(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    d = float(box_a_area + box_b_area - inter_area)
    if d == 0:
        return 0
    iou = inter_area / d
    return iou


def _detect_faces(face_driver, frame, threshold=0.4, offset=(0, 0)):
    data = cv2.resize(frame, (320, 320))
    result = face_driver.predict({'frame': data})
    count = result[3][0]
    h = frame.shape[0]
    w = frame.shape[1]

    boxes = []
    for i in range(int(round(count))):
        score = result[2][i]
        if score > threshold:
            y1 = max(0, int(result[0][4 * i] * h)) + offset[1]
            x1 = max(0, int(result[0][4 * i + 1] * w)) + offset[0]
            y2 = min(h, int(result[0][4 * i + 2] * h)) + offset[0]
            x2 = min(w, int(result[0][4 * i + 3] * w)) + offset[0]
            boxes.append((x1, y1, x2, y2))
    return boxes


def process(inputs, ctx, **kwargs):
    frame, _ = helpers.load_image(inputs, 'input')
    left = frame.copy()
    h = frame.shape[0]
    w = frame.shape[1]
    face_driver = ctx.drivers[0]
    boxes = _detect_faces(face_driver, frame, treshhold)

    def add_box(b):
        for b0 in boxes:
            if box_intersection(b0, b) > 0.1:
                return
        boxes.append(b)

    for split_count in split_counts:
        size_multiplier = 2. / (split_count + 1)
        xstep = int(frame.shape[1] / (split_count + 1))
        ystep = int(frame.shape[0] / (split_count + 1))

        xlimit = int(np.ceil(frame.shape[1] * (1 - size_multiplier)))
        ylimit = int(np.ceil(frame.shape[0] * (1 - size_multiplier)))
        for x in range(0, xlimit, xstep):
            for y in range(0, ylimit, ystep):
                y_border = min(frame.shape[0], int(np.ceil(y + frame.shape[0] * size_multiplier)))
                x_border = min(frame.shape[1], int(np.ceil(x + frame.shape[1] * size_multiplier)))
                crop = frame[y:y_border, x:x_border, :]

                box_candidates = _detect_faces(face_driver, crop, treshhold, (x, y))

                for b in box_candidates:
                    add_box(b)

    for b in boxes:
        left = cv2.rectangle(left, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), thickness=2)
    frame = np.concatenate([frame, left], axis=1)
    frame = cv2.resize(frame, (int(w), int(h / 2)))
    return {'output': frame}
