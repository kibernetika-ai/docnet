import cv2
import time
import numpy as np
import logging
import json
import base64

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', ':', '(', ')', '.', ',', '/'
    # Apostrophe only for specific cases (eg. : O'clock)
                                  "'",
    " ",
    # "end of sentence" character for CTC algorithm
    '_'
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


chrset_index = {}


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    charset, _ = read_charset()
    global chrset_index
    chrset_index = charset
    LOG.info("Init hooks")


def preprocess_boxes(inputs, ctx):
    image = inputs['image'][0]
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    w = image.shape[1]
    h = image.shape[0]
    if w > h:
        if w > 1280:
            h = int(float(h) * 1280.0 / float(w))
            w = 1280
    else:
        if ctx.h > 1280:
            w = int(w * 1280.0 / float(h))
            h = 1280
    image = cv2.resize(image, (w, h))
    ctx.image = image
    image = cv2.resize(image, (1280, 768))
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, 0)
    return {
        'Placeholder': image,
    }


def softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p


def findRoot(point, group_mask):
    root = point
    update_parent = False
    stop_loss = 1000
    while group_mask[root] != -1:
        root = group_mask[root]
        update_parent = True;
        stop_loss -= 1
        if stop_loss < 0:
            raise Exception('Stop loss')
    if update_parent:
        group_mask[point] = root
    return root


def join(p1, p2, group_mask):
    root1 = findRoot(p1, group_mask)
    root2 = findRoot(p2, group_mask)
    if root1 != root2:
        group_mask[root1] = root2


def get_all(points, w, h, group_mask):
    root_map = {}
    mask = np.zeros((h, w), np.int32)
    for i in range(len(points[0])):
        point_root = findRoot(points[1][i] + points[0][i] * w, group_mask)
        if root_map.get(point_root, None) is None:
            root_map[point_root] = len(root_map) + 1
        mask[points[0][i], points[1][i]] = root_map[point_root]
    return mask


def decodeImageByJoin(cls, links, cls_threshold, link_threshold):
    h = cls.shape[0]
    w = cls.shape[1]
    pixel_mask = cls > cls_threshold
    print(pixel_mask.shape)
    link_mask = links > link_threshold
    print(link_mask.shape)
    y, x = np.where(pixel_mask == True)
    group_mask = {}
    for i in range(len(x)):
        if pixel_mask[y[i], x[i]]:
            group_mask[y[i] * w + x[i]] = -1
    for i in range(len(x)):
        neighbour = 0
        for ny in range(y[i] - 1, y[i] + 2):
            for nx in range(x[i] - 1, x[i] + 2):
                if nx == x[i] and ny == y[i]:
                    continue
                if nx >= 0 and nx < w and ny >= 0 and ny < h:
                    pixel_value = pixel_mask[ny, nx]
                    link_value = link_mask[ny, nx, neighbour]
                    if pixel_value and link_value:
                        join(y[i] * w + x[i], ny * w + nx, group_mask)
                neighbour += 1
    return get_all((y, x), w, h, group_mask)


def maskToBoxes(mask, image_size):
    bboxes = []
    min_val, max_val, _, _ = cv2.minMaxLoc(mask)
    print(max_val)
    print(mask.shape)
    if max_val > 100:
        return max_val
    resized_mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    for i in range(int(max_val)):
        bbox_mask = resized_mask == (i + 1)
        bbox_mask = bbox_mask.astype(np.int32)
        contours, _ = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            continue
        r = cv2.minAreaRect(contours[0])
        bboxes.append(r)
    return bboxes


def postprocess_boxes(outputs, ctx):
    cls = np.transpose(outputs['pixel_cls/add_2'][0], [1, 2, 0])
    links = np.transpose(outputs['pixel_link/add_2'][0], [1, 2, 0])
    cls = softmax(cls, axis=2)
    links = np.reshape(links, (links.shape[0], links.shape[1], int(links.shape[2] / 2), 2))
    links = softmax(links, axis=3)
    mask = decodeImageByJoin(cls[:, :, 1], links[:, :, :, 1], 0.4, 0.4)
    bboxes = maskToBoxes(mask, (ctx.image.shape[1], ctx.image.shape[0]))
    outimage = ctx.image
    for i in bboxes:
        box = cv2.boxPoints(i)
        box = np.int0(box)
        outimage = cv2.drawContours(outimage, [box], 0, (0, 0, 255), 2)

    to_predict = []
    outimages = []
    outscores = []
    for i in range(len(bboxes)):
        cmask = np.zeros((ctx.image.shape[0], ctx.image.shape[1], 3), np.float32)
        box = np.int0(cv2.boxPoints(bboxes[i]))
        mask = cv2.drawContours(cmask, [box], 0, (1, 1, 1), -1)
        mask = ctx.image * mask
        maxp = np.max(box, axis=0)
        minp = np.min(box, axis=0)
        text_img = mask[minp[1]:maxp[1], minp[0]:maxp[0], :]
        _, buf = cv2.imencode('.png', text_img)
        buf = np.array(buf).tostring()
        encoded = base64.encodebytes(buf).decode()
        outimages.append(encoded)
        outscores.append(1.0)
        text_img = text_img[:,:,::-1]
        text_img = norm_image_for_text_prediction(text_img, 32, 320)
        to_predict.append(np.expand_dims(text_img, 0))

    ctx.image = outimage
    ctx.outscores = outscores
    ctx.outimages = outimages
    for i in to_predict:
        yield {
            'images': i,
        }


def final_postprocess(outputs_it, ctx):
    n = 0
    table = []
    for outputs in outputs_it:
        predictions = outputs['output']
        line = []
        end_line = len(chrset_index) - 1
        for i in predictions[0]:
            if i == end_line:
                break;
            t = chrset_index.get(i, -1)
            if t == -1:
                continue;
            line.append(t)
        line = ''.join(line)
        table.append(
            {
                'type': 'text',
                'name': line,
                'prob': float(ctx.outscores[n]),
                'image': ctx.outimages[n]
            }
        )
        n += 1
    _, buf = cv2.imencode('.png', ctx.image)
    image = np.array(buf).tostring()
    table = json.dumps(table)
    return {
        'output': image,
        'table_output': table,
    }


def norm_image_for_text_prediction(im, infer_height, infer_width):
    w = im.shape[1]
    h = im.shape[0]
    ration_w = max(w / infer_width, 1.0)
    ration_h = max(h / infer_height, 1.0)
    ratio = max(ration_h, ration_w)
    if ratio > 1:
        width = int(w / ratio)
        height = int(h / ratio)
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 127.5 - 1
    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im


preprocess = [preprocess_boxes, None]
postprocess = [postprocess_boxes, final_postprocess]
