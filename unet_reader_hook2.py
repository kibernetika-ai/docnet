import cv2
import unet.util as util
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
    image = cv2.resize(image[:,:,::-1], (w, h))
    ctx.image = image
    image = cv2.resize(image, (320, 320))
    image = image.astype(np.float32)/255.0
    image = np.expand_dims(image, 0)
    return {
        'image': image,
    }


def postprocess_boxes(outputs, ctx):
    cls = outputs['pixel_pos_scores'][0]
    links = outputs['link_pos_scores'][0]
    mask = decode_image_by_join(cls, links, 0.4, 0.4)
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
        text_img = mask[minp[1]:maxp[1], minp[0]:maxp[0], :]
        _, buf = cv2.imencode('.png', text_img[:,:,::-1])
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
    _, buf = cv2.imencode('.png', ctx.image[:,:,::-1])
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

def decode_image_by_join(pixel_scores, link_scores,
                         pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = zip(*np.where(pixel_mask))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)
    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)

        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(pixel_mask, dtype = np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    # join by link
    for point in points:
        y, x = point
        neighbours = util.get_neighbours(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if util.is_valid_cord(nx, ny, w, h):
                #                 reversed_neighbours = get_neighbours(nx, ny)
                #                 reversed_idx = reversed_neighbours.index((x, y))
                link_value = link_mask[y, x, n_idx]# and link_mask[ny, nx, reversed_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx))

    mask = get_all()
    return mask

preprocess = [preprocess_boxes, None]
postprocess = [postprocess_boxes, final_postprocess]
