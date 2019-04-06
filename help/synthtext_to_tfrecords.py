#encoding=utf-8
import tensorflow as tf
import argparse
import logging
import numpy as np
import os
import cv2
import scipy.io as sio

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_example(image_data, filename, labels,labels_text, bboxes, oriented_bboxes, shape):
    image_format = b'JPEG'
    oriented_bboxes = np.asarray(oriented_bboxes)
    if len(bboxes) == 0:
        logging.info('{} has no bboxes'.format(filename))

    bboxes = np.asarray(bboxes)
    def get_list(obj, idx):
        if len(obj) > 0:
            return list(obj[:, idx])
        return []
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/shape': int64_feature(list(shape)),
        'image/object/bbox/xmin': float_feature(get_list(bboxes, 0)),
        'image/object/bbox/ymin': float_feature(get_list(bboxes, 1)),
        'image/object/bbox/xmax': float_feature(get_list(bboxes, 2)),
        'image/object/bbox/ymax': float_feature(get_list(bboxes, 3)),
        'image/object/bbox/x1': float_feature(get_list(oriented_bboxes, 0)),
        'image/object/bbox/y1': float_feature(get_list(oriented_bboxes, 1)),
        'image/object/bbox/x2': float_feature(get_list(oriented_bboxes, 2)),
        'image/object/bbox/y2': float_feature(get_list(oriented_bboxes, 3)),
        'image/object/bbox/x3': float_feature(get_list(oriented_bboxes, 4)),
        'image/object/bbox/y3': float_feature(get_list(oriented_bboxes, 5)),
        'image/object/bbox/x4': float_feature(get_list(oriented_bboxes, 6)),
        'image/object/bbox/y4': float_feature(get_list(oriented_bboxes, 7)),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/format': bytes_feature(image_format),
        'image/filename': bytes_feature(bytes(filename, "utf8")),
        'image/encoded': bytes_feature(image_data)}))
    return example

class SynthTextDataFetcher():
    def __init__(self, mat_path, root_path):
        self.mat_path = mat_path
        self.root_path = root_path
        self._load_mat()


    def _load_mat(self):
        data = sio.loadmat(self.mat_path)
        self.image_paths = data['imnames'][0]
        self.image_bbox = data['wordBB'][0]
        self.txts = data['txt'][0]
        self.num_images =  len(self.image_paths)

    def get_image_path(self, idx):
        image_path = os.path.join(self.root_path, self.image_paths[idx][0])
        return image_path

    def get_num_words(self, idx):
        try:
            return np.shape(self.image_bbox[idx])[2]
        except: # error caused by dataset
            return 1


    def get_word_bbox(self, img_idx, word_idx):
        boxes = self.image_bbox[img_idx]
        if len(np.shape(boxes)) ==2: # error caused by dataset
            boxes = np.reshape(boxes, (2, 4, 1))

        xys = boxes[:,:, word_idx]
        assert(np.shape(xys) ==(2, 4))
        return np.float32(xys)

    def normalize_bbox(self, xys, width, height):
        xs = xys[0, :]
        ys = xys[1, :]

        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        # bound them in the valid range
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width, max_x)
        max_y = min(height, max_y)

        # check the w, h and area of the rect
        w = max_x - min_x
        h = max_y - min_y
        is_valid = True

        if w < 10 or h < 10:
            is_valid = False

        if w * h < 100:
            is_valid = False

        xys[0, :] = xys[0, :] / width
        xys[1, :] = xys[1, :] / height

        return is_valid, min_x / width, min_y /height, max_x / width, max_y / height, xys

    def get_txt(self, image_idx, word_idx):
        txts = self.txts[image_idx]
        clean_txts = []
        for txt in txts:
            clean_txts += txt.split()
        return str(clean_txts[word_idx])


    def fetch_record(self, image_idx):
        image_path = self.get_image_path(image_idx)
        if not (os.path.exists(image_path)):
            return None
        img = cv2.imread(image_path)
        h, w = img.shape[0:-1]
        num_words = self.get_num_words(image_idx)
        rect_bboxes = []
        full_bboxes = []
        txts = []
        for word_idx in range(num_words):
            xys = self.get_word_bbox(image_idx, word_idx)
            is_valid, min_x, min_y, max_x, max_y, xys = self.normalize_bbox(xys, width = w, height = h)
            if not is_valid:
                continue
            rect_bboxes.append([min_x, min_y, max_x, max_y])
            xys = np.reshape(np.transpose(xys), -1)
            full_bboxes.append(xys)
            txt = self.get_txt(image_idx, word_idx)
            txts.append(txt)
        if len(rect_bboxes) == 0:
            return None

        return image_path, img, txts, rect_bboxes, full_bboxes



def convert(image_idxes,fetcher,out_path , records_per_file = 50000):
    record_count = 0
    tfrecord_writer = None
    for image_idx in image_idxes:
        if record_count % records_per_file == 0:
            fid = int(record_count / records_per_file)
            if tfrecord_writer is not None:
                tfrecord_writer.close()
            tfrecord_writer = tf.python_io.TFRecordWriter(os.path.join(out_path,'{}.record'.format(fid)))

        record = fetcher.fetch_record(image_idx)
        if record is None:
            logging.info('image {} does not exist'.format(image_idx + 1))
            continue
        record_count += 1
        image_path, image, txts, rect_bboxes, oriented_bboxes = record
        labels = []
        labels_text = []
        for txt in txts:
            if len(txt) < 2:
                labels.append(-1)
            else:
                labels.append(1)
        labels_text.append(bytes(txt, "utf8"))
        with open(image_path,'rb') as f:
            image_data = f.read()
        shape = image.shape
        image_name = os.path.basename(image_path).split('.')[0]
        example = convert_to_example(image_data, image_name, labels,labels_text,rect_bboxes, oriented_bboxes, shape)
        tfrecord_writer.write(example.SerializeToString())
    if tfrecord_writer is not None:
        tfrecord_writer.close()

def cvt_to_tfrecords(train_path,test_path ,test,data_path, gt_path, records_per_file = 50000):

    fetcher = SynthTextDataFetcher(root_path = data_path, mat_path = gt_path)
    image_indexes = [i for i in range(fetcher.num_images)]
    np.random.shuffle(image_indexes)
    test_count = int(len(image_indexes)/100*test)
    train_count = len(image_indexes)-test_count
    train_indxes = image_indexes[:train_count]
    test_indexes = image_indexes[train_count:]
    convert(train_indxes,fetcher,train_path , records_per_file)
    convert(test_indexes,fetcher,test_path, records_per_file)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_path', type=str)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test', type=int, default=10)
    return parser

if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    test_dir = os.path.join(args.output_path,'test')
    train_dir = os.path.join(args.output_path,'train')
    if not tf.gfile.Exists(test_dir):
        tf.gfile.MakeDirs(test_dir)
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MakeDirs(train_dir)

    cvt_to_tfrecords(train_dir,test_dir,args.test,data_path = args.root_path, gt_path = args.mat_path)

