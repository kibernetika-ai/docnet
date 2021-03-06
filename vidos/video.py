import cv2
import argparse
import time
from ml_serving.drivers import driver
import numpy as np
import math
import fuzzyset
import threading


lock = threading.Lock()
ENGLISH_CHAR_MAP1 = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0','1','2','3','4','5','6','7','8','9',
    '-',':','(',')','.',',','/','$',
    "'",
    " ",
    '_'
]

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

def get_parser():
    parser = argparse.ArgumentParser(
        description='Test background'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=1024,
        help='Image size',
    )
    parser.add_argument(
        '--adjust',
        type=int,
        default=32,
        help='Adjsut',
    )
    parser.add_argument(
        '--camera',
        type=str,
        help='Full URL to network camera.',
    )
    parser.add_argument('--model')
    return parser


def add_overlays(frame, score,text):
    cv2.putText(frame,'{}: {}'.format(text,score), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)


def fix_length(l,b):
    return int(math.ceil(l/b)*b)

def get_text(labels):
    line = []
    end_line = len(chrset_index)-1
    for i in labels:
        if i == end_line:
            break
        t = chrset_index.get(i, -1)
        if t == -1:
            continue
        line.append(t)
    return ''.join(line)

def choose_one(names,candidates):
    a = []
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i==j:
                continue
            full_name = candidates[i]+' '+candidates[j]
            e = names.get(full_name)
            if (e is not None) and len(e)>0:
                print(e[0])
                a.append(e[0])
    if len(a)<1:
        return None
    a.sort(key=lambda tup: tup[0])
    a = a[len(a)-1]
    if a[0]<0.6:
        return None
    return a

to_process = None
result = None
last_processed = None
runned = True
def process():
    size = 1024
    charset, _ = read_charset()
    global chrset_index
    chrset_index = charset
    names = fuzzyset.FuzzySet()
    names.add('stas khirman')
    names.add('khirman stas')
    names.add('stas')
    names.add('khirman')
    drv1 = driver.load_driver('tensorflow')
    serving1 = drv1()
    serving1.load_model('./m1')
    drv2 = driver.load_driver('tensorflow')
    serving2 = drv2()
    serving2.load_model('./m2')
    global to_process
    i_name = 1
    while runned:
        lock.acquire(blocking=True)
        frame = to_process
        if frame is None:
            lock.release()
            continue
        print('start frame')
        to_process = None
        w = frame.shape[1]
        h = frame.shape[0]
        if w > h:
            if w > size:
                ratio = size / float(w)
                h = int(float(h) * ratio)
                w = size
            else:
                if h > size:
                    ratio = size / float(h)
                    w = int(float(w) * ratio)
                    h = size
        w = fix_length(w,32)
        h = fix_length(h,32)
        original = frame[:, :, ::-1].copy()
        image = cv2.resize(original, (w, h))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, 0)
        outputs = serving1.predict({'image': image})
        cls = outputs['pixel_pos_scores'][0]
        links = outputs['link_pos_scores'][0]
        mask = decodeImageByJoin(cls, links, 0.5, 0.1)
        bboxes = maskToBoxes(mask, (original.shape[1], original.shape[0]))
        found_name = None
        candidates = []
        for i in range(len(bboxes)):
            box = np.int0(cv2.boxPoints(bboxes[i]))
            maxp = np.max(box, axis=0) + 2
            minp = np.min(box, axis=0) - 2

            y1 = max(0, minp[1])
            y2 = min(original.shape[0], maxp[1])
            x1 = max(0, minp[0])
            x2 = min(original.shape[1], maxp[0])
            text_img = original[y1:y2, x1:x2, :]
            if text_img.shape[0] < 4 or text_img.shape[1] < 4:
                continue
            #if bboxes[i][1][0]>bboxes[i][1][1]:
            #    angle = -1*bboxes[i][2]
            #else:
            #    angle = -1*(90+bboxes[i][2])
            #if angle!=0:
            #    text_img = rotate_bound(text_img,angle)
            text_img = norm_image_for_text_prediction(text_img, 32, 320)
            text_img = np.expand_dims(text_img, 0)
            text = serving2.predict({'images':text_img})
            text = text['output'][0]
            text = get_text(text)
            if len(text)>2:
                print('text: {}'.format(text))
                found = names.get(text)
                if (found is not None) and (len(found)>0):
                    print(found[0])
                    if found[0][0]>0.7:
                        text = found[0][1]
                        if ' ' in text:
                            found_name = (found[0][0],text)
                            candidates = []
                            break
                        else:
                            candidates.append(text)
            if (found_name is None) and len(candidates)>0:
                found_name = choose_one(names,candidates)
        for i in bboxes:
            box = cv2.boxPoints(i)
            box = np.int0(box)
            original = cv2.drawContours(original, [box], 0, (255, 0, 0), 2)
        frame = np.ascontiguousarray(original[:, :, ::-1],np.uint8)
        if found_name is not None:
            add_overlays(frame,found_name[0],found_name[1])
            cv2.imwrite('results/result_{}.jpg'.format(i_name),frame)
            global result
            result = frame
            i_name+=1
        global last_processed
        last_processed = frame
        lock.release()
        print('stop frame')

def make_small(frame,size):
    w = frame.shape[1]
    h = frame.shape[0]
    if w > h:
        if w > size:
            ratio = size / float(w)
            h = int(float(h) * ratio)
            w = size
        else:
            if h > size:
                ratio = size / float(h)
                w = int(float(w) * ratio)
                h = size
    return cv2.resize(frame, (w, h))

def main():

    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()
    if args.camera:
        video_capture = cv2.VideoCapture(args.camera)
    else:
        video_capture = cv2.VideoCapture(0)
    p = threading.Thread(target=process)
    p.start()
    local_result = None
    last_result = None
    show_size = 768
    try:
        doit = True

        while doit:
            _, frame = video_capture.read()
            sframe = frame.copy()
            f1 = make_small(sframe,show_size)
            if local_result is None:
                local_result = make_small(sframe,show_size//2)
            if last_result is None:
                last_result = make_small(sframe,show_size//2)
            f2 = np.concatenate([last_result,local_result],axis=1)
            f1 = np.concatenate([f1,f2],axis=0)
            cv2.imshow('Video', f1)
            if lock.acquire(blocking=False):
                global to_process
                to_process=frame
                if result is not None:
                    local_result = make_small(result,show_size//2)
                if last_processed is not None:
                    last_result = make_small(last_processed,show_size//2)
                    #cv2.imshow('Video', frame)
                lock.release()
            key = cv2.waitKey(1)
            # Wait 'q' or Esc
            if key == ord('q') or key == 27:
                break

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))
    global runned
    runned = False
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print('Finished')


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
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_CUBIC)
    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im

def norm_image_for_text_prediction0(im, infer_height, infer_width):
    w = im.shape[1]
    h = im.shape[0]
    ration_w = max(w / infer_width, 1.0)
    ration_h = max(h / infer_height, 1.0)
    ratio = max(ration_h, ration_w)
    if ratio > 1:
        width = int(w / ratio)
        height = int(h / ratio)
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 255.0
    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im

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
        contours = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        if len(contours) < 1:
            continue
        maxarea = 0
        maxc = None
        for j in contours:
            if len(j)>1:
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


if __name__ == "__main__":
    main()
