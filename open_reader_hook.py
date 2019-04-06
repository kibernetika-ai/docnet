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

    cmask = np.zeros((ctx.image.shape[0], ctx.image.shape[1], 3), np.float32)
    to_predict = []
    outimages = []
    outscores = []
    for i in range(len(bboxes)):
        box = np.int0(cv2.boxPoints(bboxes[i]))
        mask = cv2.drawContours(cmask, [box], 0, (1, 1, 1), -1)
        mask = ctx.image * mask
        maxp = np.max(box, axis=0)
        minp = np.min(box, axis=0)
        text_img = mask[minp[1]:maxp[1], minp[0]:maxp[0], ::-1]
        _, buf = cv2.imencode('.png', text_img)
        buf = np.array(buf).tostring()
        encoded = base64.encodebytes(buf).decode()
        outimages.append(encoded)
        outscores.append(1.0)
        text_img = norm_image_for_text_prediction(text_img, 32, 320)
        to_predict.append(np.expand_dims(text_img, 0))

    ctx.image = outimage[:, :, ::-1]
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


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle_rbox(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    boxes = nms_locality(boxes.astype('float32'), nms_thres)
    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def resize_image(im, max_side_len=2400):
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def intersection(g, p):
    g = g[:8].reshape((4, 2))
    p = p[:8].reshape((4, 2))
    g_max = np.max(g, axis=0)
    g_min = np.min(g, axis=0)
    p_max = np.max(p, axis=0)
    p_min = np.min(p, axis=0)
    g_area = (g_max[0] - g_min[0]) * (g_max[1] - g_min[1])
    p_area = (p_max[0] - p_min[0]) * (p_max[1] - p_min[1])
    ymin = max(g_min[1], p_min[1])
    xmin = max(g_min[0], p_min[0])
    ymax = min(g_max[1], p_max[1])
    xmax = min(g_max[0], p_max[0])
    h = max(ymax - ymin, 0.)
    w = max(xmax - xmin, 0.)
    inter = h * w
    union = g_area + p_area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


preprocess = [preprocess_boxes, None]
postprocess = [postprocess_boxes, final_postprocess]
