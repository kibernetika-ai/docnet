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
        if h > 1280:
            w = int(w * 1280.0 / float(h))
            h = 1280
    image = cv2.resize(image[:, :, ::-1], (w, h))
    ctx.image = image
    image = cv2.resize(image, (320, 320))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, 0)
    return {
        'image': image,
    }


def findRoot(point, group_mask):
    root = point
    update_parent = False
    stop_loss = 1000
    while group_mask[root] != -1:
        root = group_mask[root]
        update_parent = True
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
    link_mask = links > link_threshold
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


def maskToBoxes(mask, image_size, min_area=300, min_height=10):
    bboxes = []
    min_val, max_val, _, _ = cv2.minMaxLoc(mask)
    resized_mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    for i in range(int(max_val)):
        bbox_mask = resized_mask == (i + 1)
        bbox_mask = bbox_mask.astype(np.int32)
        contours, _ = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            continue
        r = cv2.minAreaRect(contours[0])

        if min(r[1][0], r[1][1]) < min_height:
            continue
        if r[1][0] * r[1][1] < min_area:
            continue
        bboxes.append(r)
    return bboxes

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def postprocess_boxes(outputs, ctx):
    cls = outputs['pixel_pos_scores'][0]
    links = outputs['link_pos_scores'][0]
    #testmask = cv2.resize(cls, (ctx.image.shape[1], ctx.image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = decodeImageByJoin(cls, links, 0.5, 0)
    bboxes = maskToBoxes(mask, (ctx.image.shape[1], ctx.image.shape[0]))
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
        y1 = max(0,minp[1])
        y2 = min(ctx.image.shape[0],maxp[1])
        x1 = max(0,minp[0])
        x2 = min(ctx.image.shape[1],maxp[0])
        text_img = mask[y1:y2, x1:x2, :]
        if text_img.shape[0] < 1 or text_img.shape[1]<1:
            logging.info('Skip box: {}'.format(box))
            continue
        text_img = rotate_bound(text_img,-1*bboxes[i][2])
        _, buf = cv2.imencode('.png', text_img[:, :, ::-1])
        buf = np.array(buf).tostring()
        encoded = base64.encodebytes(buf).decode()
        outimages.append(encoded)
        outscores.append(1.0)
        text_img = norm_image_for_text_prediction(text_img, 32, 320)
        to_predict.append(np.expand_dims(text_img, 0))

    for i in bboxes:
        box = cv2.boxPoints(i)
        box = np.int0(box)
        ctx.image = cv2.drawContours(ctx.image, [box], 0, (255, 0, 0), 2)
    #ctx.image = ctx.image.astype(np.float32)*np.expand_dims(testmask,2)
    #ctx.image = ctx.image.astype(np.uint8)
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
    _, buf = cv2.imencode('.png', ctx.image[:, :, ::-1])
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
