import tensorflow as tf
import numpy as np
import cv2
import unet.util as util
import glob


def tf_cal_gt_for_single_image(xs, ys, labels, resolution):
    pixel_cls_label, pixel_cls_weight, \
    pixel_link_label, pixel_link_weight = \
        tf.py_func(
            cal_gt_for_single_image,
            [xs, ys, labels, resolution],
            [tf.int32, tf.float32, tf.int32, tf.float32]
        )

    num_neighbours = 8
    pixel_cls_label.set_shape([resolution, resolution])
    pixel_cls_weight.set_shape([resolution, resolution])
    pixel_link_label.set_shape([resolution, resolution, num_neighbours])
    pixel_link_weight.set_shape([resolution, resolution, num_neighbours])
    return pixel_cls_label, pixel_cls_weight, \
           pixel_link_label, pixel_link_weight


def cal_gt_for_single_image(normed_xs, normed_ys, labels, resolution):
    pixel_cls_weight_method = 'PIXEL_CLS_WEIGHT_bbox_balanced'
    text_label = 1
    ignore_label = -1
    background_label = 0
    num_neighbours = 8
    bbox_border_width = 1
    pixel_cls_border_weight_lambda = 1.0

    # validate the args
    assert np.ndim(normed_xs) == 2
    assert np.shape(normed_xs)[-1] == 4
    assert np.shape(normed_xs) == np.shape(normed_ys)
    assert len(normed_xs) == len(labels)

    #     assert set(labels).issubset(set([text_label, ignore_label, background_label]))

    num_positive_bboxes = np.sum(np.asarray(labels) == text_label)
    # rescale normalized xys to absolute values
    xs = normed_xs * resolution
    ys = normed_ys * resolution

    # initialize ground truth values
    mask = np.zeros((resolution, resolution), dtype=np.int32)
    pixel_cls_label = np.ones((resolution, resolution), dtype=np.int32) * background_label
    pixel_cls_weight = np.zeros((resolution, resolution), dtype=np.float32)

    pixel_link_label = np.zeros((resolution, resolution, num_neighbours), dtype=np.int32)
    pixel_link_weight = np.ones((resolution, resolution, num_neighbours), dtype=np.float32)

    # find overlapped pixels, and consider them as ignored in pixel_cls_weight
    # and pixels in ignored bboxes are ignored as well
    # That is to say, only the weights of not ignored pixels are set to 1

    ## get the masks of all bboxes
    bbox_masks = []
    pos_mask = mask.copy()
    for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
        if labels[bbox_idx] == background_label:
            continue

        bbox_mask = mask.copy()

        bbox_points = zip(bbox_xs, bbox_ys)
        bbox_contours = util.points_to_contours(bbox_points)
        bbox_mask = cv2.drawContours(bbox_mask, bbox_contours, -1, 1, thickness=-1)
        bbox_masks.append(bbox_mask)

        if labels[bbox_idx] == text_label:
            pos_mask += bbox_mask

    # treat overlapped in-bbox pixels as negative,
    # and non-overlapped  ones as positive
    pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
    num_positive_pixels = np.sum(pos_mask)

    ## add all bbox_maskes, find non-overlapping pixels
    sum_mask = np.sum(bbox_masks, axis=0)
    not_overlapped_mask = sum_mask == 1

    ## gt and weight calculation
    for bbox_idx, bbox_mask in enumerate(bbox_masks):
        bbox_label = labels[bbox_idx]
        if bbox_label == ignore_label:
            # for ignored bboxes, only non-overlapped pixels are encoded as ignored
            bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
            pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
            continue

        if labels[bbox_idx] == background_label:
            continue
        # from here on, only text boxes left.

        # for positive bboxes, all pixels within it and pos_mask are positive
        bbox_positive_pixel_mask = bbox_mask * pos_mask
        # background or text is encoded into cls gt
        pixel_cls_label += bbox_positive_pixel_mask * bbox_label

        # for the pixel cls weights, only positive pixels are set to ones
        if pixel_cls_weight_method == 'PIXEL_CLS_WEIGHT_all_ones':
            pixel_cls_weight += bbox_positive_pixel_mask
        elif pixel_cls_weight_method == 'PIXEL_CLS_WEIGHT_bbox_balanced':
            # let N denote num_positive_pixels
            # weight per pixel = N /num_positive_bboxes / n_pixels_in_bbox
            # so all pixel weights in this bbox sum to N/num_positive_bboxes
            # and all pixels weights in this image sum to N, the same
            # as setting all weights to 1
            num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
            if num_bbox_pixels > 0:
                per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_bbox_pixels
                pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight
        else:
            raise ValueError('pixel_cls_weight_method not supported:{}'.format(pixel_cls_weight_method))

        ## calculate the labels and weights of links
        ### for all pixels in  bboxes, all links are positive at first
        bbox_point_cords = np.where(bbox_positive_pixel_mask)
        pixel_link_label[bbox_point_cords] = 1

        ## the border of bboxes might be distored because of overlapping
        ## so recalculate it, and find the border mask
        new_bbox_contours = util.find_contours(bbox_positive_pixel_mask)
        bbox_border_mask = mask.copy()
        util.draw_contours(bbox_border_mask, new_bbox_contours, -1,
                           color=1, border_width=bbox_border_width * 2 + 1)
        bbox_border_mask *= bbox_positive_pixel_mask
        bbox_border_cords = np.where(bbox_border_mask)

        ## give more weight to the border pixels if configured
        pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda

        ### change link labels according to their neighbour status
        border_points = zip(*bbox_border_cords)

        def in_bbox(nx, ny):
            return bbox_positive_pixel_mask[ny, nx]

        for y, x in border_points:
            neighbours = util.get_neighbours(x, y)
            for n_idx, (nx, ny) in enumerate(neighbours):
                if not util.is_valid_cord(nx, ny, resolution, resolution) or not in_bbox(nx, ny):
                    pixel_link_label[y, x, n_idx] = 0

    pixel_cls_weight = np.asarray(pixel_cls_weight, dtype=np.float32)
    pixel_link_weight *= np.expand_dims(pixel_cls_weight, axis=-1)
    return pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight


def data_fn(params, training):
    data_set = params['data_set']
    datasets_files = []
    for tf_file in glob.iglob(data_set + '/*.record'):
        datasets_files.append(tf_file)
    resolution = params['resolution']

    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)

        def _parser(example):
            features = {
                'image/encoded':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/shape':
                    tf.FixedLenFeature(3, tf.int64),
                'image/object/bbox/x1':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/x2':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/x3':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/x4':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/y1':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/y2':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/y3':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/y4':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/label':
                    tf.VarLenFeature(tf.int64)
            }
            res = tf.parse_single_example(example, features)
            img = tf.image.decode_jpeg(res['image/encoded'], channels=3)
            original_w = tf.cast(res['image/shape'][1], tf.int32)
            original_h = tf.cast(res['image/shape'][0], tf.int32)
            img = tf.reshape(img, [1, original_h, original_w, 3])
            img = tf.image.resize_images(img, [resolution, resolution])[0]
            img = tf.cast(img,tf.float32)/255.0
            x1 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x1']), tf.float32)
            x2 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x2']), tf.float32)
            x3 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x3']), tf.float32)
            x4 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x4']), tf.float32)
            y1 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y1']), tf.float32)
            y2 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y2']), tf.float32)
            y3 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y3']), tf.float32)
            y4 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y4']), tf.float32)
            gxs = tf.transpose(tf.stack([x1, x2, x3, x4]))
            gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
            labels = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/label']), tf.int32)
            pixel_cls_label, pixel_cls_weight, \
            pixel_link_label, pixel_link_weight = \
                tf_cal_gt_for_single_image(gxs, gys, labels, resolution)
            return img, pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight

        ds = ds.map(_parser)
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))

        def _features_labels(img, pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight):
            return img, {'pixel_cls_label': pixel_cls_label,
                         'pixel_cls_weight': pixel_cls_weight,
                         'pixel_link_label': pixel_link_label,
                         'pixel_link_weight': pixel_link_weight}

        ds = ds.map(_features_labels)
        if training:
            ds = ds.repeat(params['num_epochs']).prefetch(params['batch_size'])
        return ds

    return _input_fn
