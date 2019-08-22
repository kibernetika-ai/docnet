import logging

import cv2
import numpy as np
import ml_serving.utils.helpers as helpers

LOG = logging.getLogger(__name__)
PARAMS = {
    'test': 1
}


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)


def resize(img, s):
    h = img.shape[0]
    w = img.shape[1]
    if h <= s and w <= s:
        pw = s - w
        ph = s - h
        return np.pad(img, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0),w,h
    if w>h:
        h = min(int(h*s/w),s)
        w = s
        pw = 0
        ph = s-h
    else:
        w = min(int(w*s/h),s)
        h = s
        ph = 0
        pw = s-w
    img = cv2.resize(img,(w,h))
    if pw>0 or ph>0:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0),w,h
    return img




def process(inputs, ctx, **kwargs):
    frame, _ = helpers.load_image(inputs, 'input')
    data,iw,ih = resize(frame,512)
    h = frame.shape[0]
    w = frame.shape[1]
    driver = ctx.drivers[0]
    data = data.astype(np.float32) / 255.0
    data = np.expand_dims(data, 0)
    result = driver.predict({'image': data})
    cls = result['Reshape_1'][0,0:ih,0:iw,1]
    links = result['Reshape_4'][0,0:ih,0:iw,:,1]
    out_mask = cv2.resize(cls, (w,h), interpolation=cv2.INTER_NEAREST)
    frame = frame.astype(np.float32) * np.expand_dims(out_mask, 2)
    frame = frame.astype(np.uint8)
    return {'output': frame}
