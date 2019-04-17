import cv2
import numpy as np
import logging
import json
import base64
import math

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    "'",
    " ",
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


out_types = {
    'Box': 0,
    'Mask': 1,
    'Link1': 2,
    'Link2': 3,
    'Link3': 4,
    'Link4': 5,
    'Link5': 6,
    'Link6': 7,
    'Link7': 8,
    'Link8': 9,
}


def fix_length(l, b):
    return int(math.ceil(l / b) * b)


MAX_DIM = 1024.0


def find_people(image,draw_image,ctx):
    data = cv2.resize(image,(300, 300), cv2.INTER_LINEAR)
    data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
    # convert to BGR
    data = data[:, ::-1, :, :]
    data = ctx.drivers[0].predict(data)
    data = list(data.values())[0].reshape([-1, 7])
    bboxes_raw = data[data[:, 2] > 0.25]

    w = image.shape[1]
    h = image.shape[0]
    images = []
    if bboxes_raw is not None:
        for box in bboxes_raw:
            xmin = box[3] * w
            ymin = box[4] * h
            xmax = box[5] * w
            ymax = box[6] * h
            bw = xmax-xmin
            bh = ymax-ymin
            xmin = max(xmin-bw,0)
            xmax = min(xmax+bw,w)
            ymax = min(ymax+bh*4,h)
            images.append((ymin,ymax,image[ymin:ymax,xmin:xmax,:]))
            draw_image = cv2.rectangle(draw_image,(xmin,ymin),(xmax,ymax),(0,255,0), thickness=2)
    return images,draw_image

def process(inputs, ctx):
    image = inputs['image'][0]
    pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    link_threshold = float(inputs.get('link_threshold', 0.5))
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
    images,image = find_people(image.copy(),image,ctx)
    r_, buf = cv2.imencode('.png', image[:, :, ::-1])
    image = np.array(buf).tostring()
    return {
        'output': image,
    }

def preprocess_boxes(inputs, ctx):
    image = inputs['image'][0]
    ctx.pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    ctx.resolution = int(inputs.get('resolution', 320))
    ctx.link_threshold = float(inputs.get('link_threshold', 0))
    logging.info('Output type: {}'.format(inputs.get('out_type', ['Box'])[0].decode("utf-8")))
    ctx.out_type = out_types.get(inputs.get('out_type', ['Box'])[0].decode("utf-8"), 0)
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    w = image.shape[1]
    h = image.shape[0]
    ctx.ratio = 1.0
    if w > h:
        if w > MAX_DIM:
            ctx.ratio = MAX_DIM / float(w)
            h = int(float(h) * ctx.ratio)
            w = MAX_DIM
    else:
        if h > MAX_DIM:
            ctx.ratio = MAX_DIM / float(h)
            w = int(float(w) * ctx.ratio)
            h = MAX_DIM
    w = fix_length(w, 32)
    h = fix_length(h, 32)
    ctx.image = image[:, :, ::-1].copy()
    image = cv2.resize(ctx.image, (w, h))
    # ctx.image = image
    # image = cv2.resize(image, (ctx.resolution, ctx.resolution))
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
    pixel_mask = cls >= cls_threshold
    link_mask = links >= link_threshold
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


def maskToBoxes(mask, image_size, min_area=200, min_height=6):
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
        maxarea = 0
        maxc = None
        for j in contours:
            area = cv2.contourArea(j)
            if area > maxarea:
                maxarea = area
                maxc = j
        if maxc is not None and maxarea > 36:
            r = cv2.minAreaRect(maxc)
            if min(r[1][0], r[1][1]) < min_height:
                continue
            bboxes.append(r)
        # if min(r[1][0], r[1][1]) < min_height:
        #    logging.info('Skip size box {} {}'.format(r, i + 1))
        #    continue
        # if r[1][0] * r[1][1] < min_area:
        #    logging.info('Skip area box {} {}'.format(r, i + 1))
        #    continue
        # bboxes.append(r)
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
    out_mask = None
    if ctx.out_type == 1:
        out_mask = cv2.resize(cls, (ctx.image.shape[1], ctx.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # out_mask[out_mask < ctx.pixel_threshold] = 0
    elif ctx.out_type > 1:
        out_mask = cv2.resize(links[:, :, ctx.out_type - 2], (ctx.image.shape[1], ctx.image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        ##out_mask[out_mask < ctx.link_threshold] = 0

    mask = decodeImageByJoin(cls, links, ctx.pixel_threshold, ctx.link_threshold)
    bboxes = maskToBoxes(mask, (ctx.image.shape[1], ctx.image.shape[0]))
    to_predict = []
    outimages = []
    outscores = []
    for i in range(len(bboxes)):
        # cmask = np.zeros((ctx.image.shape[0], ctx.image.shape[1], 3), np.float32)
        box = np.int0(cv2.boxPoints(bboxes[i]))

        # mask = cv2.drawContours(cmask, [box], 0, (1, 1, 1), -1)
        # mask = ctx.image * mask
        maxp = np.max(box, axis=0) + 2
        minp = np.min(box, axis=0) - 2
        # maxp = maxp.astype(np.int32)
        # minp = minp.astype(np.int32)
        y1 = max(0, minp[1])
        y2 = min(ctx.image.shape[0], maxp[1])
        x1 = max(0, minp[0])
        x2 = min(ctx.image.shape[1], maxp[0])
        text_img = ctx.image[y1:y2, x1:x2, :]
        if text_img.shape[0] < 1 or text_img.shape[1] < 1:
            logging.info('Skip box: {}'.format(box))
            continue
        if bboxes[i][1][0]>bboxes[i][1][1]:
            angle = -1*bboxes[i][2]
        else:
            angle = -1*(90+bboxes[i][2])
        if angle!=0:
            text_img = rotate_bound(text_img,angle)

        text_img = norm_image_for_text_prediction(text_img, 32, 320)
        to_predict.append(np.expand_dims(text_img.astype(np.float32) / 255.0, 0))
        _, buf = cv2.imencode('.png', text_img[:, :, ::-1])
        buf = np.array(buf).tostring()
        encoded = base64.encodebytes(buf).decode()
        outimages.append(encoded)
        outscores.append(-1 * bboxes[i][2])

    if out_mask is not None:
        ctx.image = ctx.image.astype(np.float32) * np.expand_dims(out_mask, 2)
        ctx.image = ctx.image.astype(np.uint8)
    else:
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


def get_text(predictions):
    line = []
    end_line = len(chrset_index) - 1
    for i in predictions:
        if i == end_line:
            break
        t = chrset_index.get(i, -1)
        if t == -1:
            continue
        line.append(t)
    return ''.join(line)


def final_postprocess(outputs_it, ctx):
    n = 0
    table = []
    for outputs in outputs_it:
        line = ''
        for k, v in outputs.items():
            v = get_text(v[0])
            logging.info('{}: {}'.format(k, v))
            if k == '0':
                line = v

        logging.info('Best text: {}'.format(line))
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
    #ration_w = max(w / infer_width, 1.0)
    #ration_h = max(h / infer_height, 1.0)
    #ratio = max(ration_h, ration_w)
    ratio = h/infer_height
    #if ratio > 1:
    width = int(w / ratio)
    height = int(h / ratio)
    width = min(infer_width,width)
    #im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

    #im[np.greater(im,200)]=255
    #im[np.less(im,100)]=0

    #im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)

    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_CUBIC)

    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im



