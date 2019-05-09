import cv2
import argparse
from ml_serving.drivers import multimodel
import numpy as np
import fuzzyset
import threading
import face_badge
import time
import os
import subprocess

import logging

lock = threading.Lock()
logging.getLogger().setLevel('INFO')


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
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output path',
    )
    parser.add_argument(
        '--reload',
        type=str,
        default='',
        help='Reload Path',
    )
    parser.add_argument('--model')
    return parser


output_dir = ''
reload_dir = ''


def add_overlays(frame, score, text):
    cv2.putText(frame, '{}: {}'.format(text, score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2,
                lineType=2)


def choose_one(names, candidates):
    a = []
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j:
                continue
            full_name = candidates[i] + ' ' + candidates[j]
            e = names.get(full_name)
            if (e is not None) and len(e) > 0:
                print(e[0])
                a.append(e[0])
    if len(a) < 1:
        return None
    a.sort(key=lambda tup: tup[0])
    a = a[len(a) - 1]
    if a[0] < 0.6:
        return None
    return a


to_process = None
result = None
last_processed = None
runned = True
new_count = 0


def reload_classes():
    global new_count
    new_count = 1
    while runned:
        time.sleep(10)
        if new_count > 0:
            new_count = 0
            logging.info('reload')
            rresult = subprocess.check_output(['python', 'svod_rcgn/prepare.py'], cwd=reload_dir)
            logging.info(rresult)


def process():
    names = fuzzyset.FuzzySet()
    names.add('stas khirman')
    names.add('khirman stas')
    names.add('stas')
    names.add('khirman')
    # drv = driver.load_driver('multimodel')
    serving = multimodel.MultiModelDriver(init_hook=face_badge.init_hook, process=face_badge.process_internal)
    kwargs = {'ml-serving-drivers': ['openvino', 'tensorflow', 'tensorflow']}
    serving.load_model(['./vidos/faces/face-detection.xml', './vidos/m1', './vidos/m2'], **kwargs)
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
        results = serving.predict_hooks({
            'pixel_threshold': 0.5,
            'link_threshold': 0.5,
            'image': frame
        })
        frame = results['output']
        table = results['table_output']
        found_name = None
        candidates = []
        for e in table:
            text = e['name']
            if len(text) > 2:
                found = names.get(text)
                if (found is not None) and (len(found) > 0):
                    if found[0][0] > 0.7:
                        text = found[0][1]
                        if ' ' in text:
                            found_name = (found[0][0], text)
                            candidates = []
                            break
                        else:
                            candidates.append(text)
        if (found_name is None) and len(candidates) > 0:
            found_name = choose_one(names, candidates)
        if found_name is not None:
            add_overlays(frame, found_name[0], found_name[1])
            to_save = e['image'][:, :, ::-1]
            if output_dir != '':
                name = found_name[1].replace(" ", "_")
                to_dir = '{}/{}'.format(output_dir, name)
                if not os.path.exists(to_dir):
                    os.mkdir(to_dir)
                fname = '{}/auto_{}_{}.jpg'.format(to_dir, int(time.time()), i_name)
                logging.info('Save new picture: {}'.format(fname))
                cv2.imwrite(fname, to_save)
                global new_count
                new_count = 1
            global result
            result = frame
            i_name += 1
        global last_processed
        last_processed = frame
        lock.release()
        print('stop frame')


def make_small(frame, size):
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
    parser = get_parser()
    args = parser.parse_args()
    real_sence = False
    if args.camera:
        if args.camera== "realsense":
            import pyrealsense2 as rs
            real_sence = True
            video_capture = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8,6)
            video_capture.start(config)
        else:
            video_capture = cv2.VideoCapture(args.camera)
    else:
        video_capture = cv2.VideoCapture(0)

    global output_dir
    output_dir = args.output
    global reload_dir
    reload_dir = args.reload
    p = threading.Thread(target=process)
    p.start()
    r = threading.Thread(target=reload_classes)
    r.start()
    local_result = None
    last_result = None
    show_size = 768
    try:
        doit = True

        while doit:
            if real_sence:
                frame = video_capture.wait_for_frames()
                frame = frame.get_color_frame()
                frame = np.asanyarray(frame.get_data())
            else:
                _, frame = video_capture.read()
            sframe = frame.copy()
            f1 = make_small(sframe, show_size)
            if local_result is None:
                local_result = make_small(sframe, show_size // 2)
            if last_result is None:
                last_result = make_small(sframe, show_size // 2)
            f2 = np.concatenate([last_result, local_result], axis=1)
            f1 = np.concatenate([f1, f2], axis=0)
            cv2.imshow('Video', f1)
            if lock.acquire(blocking=False):
                global to_process
                to_process = frame
                if result is not None:
                    local_result = make_small(result, show_size // 2)
                if last_processed is not None:
                    last_result = make_small(last_processed, show_size // 2)
                    # cv2.imshow('Video', frame)
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
    if real_sence:
        video_capture.stop()
    else:
        video_capture.release()
    cv2.destroyAllWindows()
    print('Finished')


if __name__ == "__main__":
    main()
