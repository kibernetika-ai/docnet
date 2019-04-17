import cv2
import argparse
import time
from ml_serving.drivers import driver
import numpy as np
import math
import fuzzyset
import threading

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
        help='Full URL to network camera.',
    )
    parser.add_argument('--model')
    return parser

def main():

    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()
    video_capture = cv2.VideoCapture("rtsp://admin:admin@192.168.1.83")
    #else:
    #    video_capture = cv2.VideoCapture(0)
    try:
        doit = True
        while doit:
            ok, frame = video_capture.read()
            if ok:
                cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
                # Wait 'q' or Esc
            if key == ord('q') or key == 27:
                break

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print('Finished')

if __name__ == "__main__":
    main()
