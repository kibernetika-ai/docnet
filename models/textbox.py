import tensorflow as tf
import numpy as np
import math
import util.textbox_common
from util import ssd_vgg_preprocessing
import logging

def conv2d(input, outputNum, kernel=[3, 3], strides=[1, 1], padding='SAME', bn=False, trainPhase=True, name='conv2d'):
    with tf.name_scope(name) as scope:
        output = tf.layers.conv2d(input, outputNum, kernel, strides=strides, padding=padding, activation=tf.nn.relu)
        if bn:
            output = tf.layers.batch_normalization(output, center=False, training=trainPhase)
        return output


def max_pool(input, stride=2, kernel=2, name='pool'):
    return tf.nn.max_pool(input, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME',
                          name=name)


def abs_smooth(x):
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def smooth_l1(x):
    l2 = 0.5 * tf.square(x)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    re = tf.where(condition, l2, l1)
    return re


def _ssd_losses(logits, localisations,
                glocalisations, gscores,
                match_threshold=0.1,
                negative_ratio=3.,
                alpha=1.,
                label_smoothing=0.,
                scope=None):
    with tf.name_scope(scope, 'text_loss'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        sum_loss = []
        for i in range(len(logits)):
            dtype = logits[i].dtype
            logging.info('logists: {}'.format(logits[i].shape))
            logging.info('gscores: {}'.format(gscores[i].shape))
            logging.info('localisations: {}'.format(localisations[i].shape))
            logging.info('glocalisations: {}'.format(glocalisations[i].shape))
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                ipmask = tf.cast(pmask, tf.int32)
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)

                # Negative mask
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)

                nvalues = tf.where(tf.cast(1 - ipmask, tf.bool), gscores[i], np.zeros(gscores[i].shape))
                nvalues_flat = tf.reshape(nvalues, [-1])
                val, idxes = tf.nn.top_k(nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = nvalues > minval
                fnmask = tf.cast(nmask, dtype)
                inmask = tf.cast(nmask, tf.int32)
                # Add cross-entropy loss.

                with tf.name_scope('cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=ipmask)
                    loss = tf.losses.compute_weighted_loss(loss, fpmask)
                    l_cross_pos.append(loss)
                    sum_loss.append(loss)

                with tf.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=inmask)
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)
                    sum_loss.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)
                    sum_loss.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')
            sum_loss = tf.add_n(l_loc, 'sum_loss')
            tf.summary.scalar('loss/cross_entropy_pos', total_cross_pos)
            tf.summary.scalar('loss/cross_entropy_neg', total_cross_neg)
            tf.summary.scalar('loss/cross_entropy', total_cross)
            tf.summary.scalar('loss/localization', total_loc)
    return sum_loss


def _text_multibox_layer(inputs,
                         trainPhase,
                         is_last=False,
                         normalization=-1):
    net = inputs
    if normalization > 0:
        net = tf.layers.batch_normalization(net, center=False, training=trainPhase)
    num_anchors = 6
    num_classes = 2
    num_loc_pred = 2 * num_anchors * 4

    if is_last:
        loc_pred = tf.layers.conv2d(net, num_loc_pred, [1, 1], activation=None, padding='VALID',
                                    name='conv_loc')
    else:
        loc_pred = tf.layers.conv2d(net, num_loc_pred, [1, 5], activation=None, padding='SAME',
                                    name='conv_loc')
    loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [2, num_anchors, 4])
    # Class prediction.
    scores_pred = 2 * num_anchors * num_classes
    if is_last:
        sco_pred = tf.layers.conv2d(net, scores_pred, [1, 1], activation=None, padding='VALID',
                                    name='conv_cls')
    else:
        sco_pred = tf.layers.conv2d(net, scores_pred, [1, 5], activation=None, padding='SAME',
                                    name='conv_cls')
    sco_pred = tf.reshape(sco_pred, sco_pred.get_shape().as_list()[:-1] + [2, num_anchors, num_classes])
    logging.info('sco_pred: {}'.format(sco_pred.shape))
    return sco_pred, loc_pred


def _box_fn(features, labels, mode, params=None, config=None):
    global_step = tf.train.get_or_create_global_step()
    input = features['image']
    glocalisations = list(features['glocalisations'])
    gscores = list(features['gscores'])

    with tf.name_scope(name="vgg") as scope:
        conv1_1 = conv2d(input, 64, name='conv1_1')  # 300
        conv1_2 = conv2d(conv1_1, 64, name='conv1_2')  # 300
        pool1 = max_pool(conv1_2, name='pool1')  # 150
        conv2_1 = conv2d(pool1, 128, name='conv2_1')  # 150
        conv2_2 = conv2d(conv2_1, 128, name='conv2_2')  # 150
        pool2 = max_pool(conv2_2, name='pool2')  # 75
        conv3_1 = conv2d(pool2, 256, name='conv3_1')  # 75
        conv3_2 = conv2d(conv3_1, 256, name='conv3_2')  # 75
        conv3_3 = conv2d(conv3_2, 256, name='conv3_3')  # 75
        pool3 = max_pool(conv3_3, name='pool3')  # 38
        conv4_1 = conv2d(pool3, 512, name='conv4_1')  # 38
        conv4_2 = conv2d(conv4_1, 512, name='conv4_2')  # 38
        conv4_3 = conv2d(conv4_2, 512, name='conv4_3')  # 38
        pool4 = max_pool(conv4_3, name='pool4')  # 19
        conv5_1 = conv2d(pool4, 512, name='conv5_1')  # 19
        conv5_2 = conv2d(conv5_1, 512, name='conv5_2')  # 19
        conv5_3 = conv2d(conv5_2, 512, name='conv5_3')  # 19
        pool5 = max_pool(conv5_3, stride=1, kernel=3, name='pool5')  # 19
        conv6 = conv2d(pool5, 1024, name='conv6')  # 19
        conv7 = conv2d(conv6, 1024, kernel=[1, 1], name='conv7')  # 19
        conv8_1 = conv2d(conv7, 256, kernel=[1, 1], name='conv8_1')  # 19
        conv8_2 = conv2d(conv8_1, 512, strides=[2, 2], name='conv8_2')  # 10
        conv9_1 = conv2d(conv8_2, 128, kernel=[1, 1], name='conv9_1')  # 10
        conv9_2 = conv2d(conv9_1, 256, strides=[2, 2], name='conv9_2')  # 5
        conv10_1 = conv2d(conv9_2, 128, kernel=[1, 1], name='conv10_1')  # 5
        conv10_2 = conv2d(conv10_1, 256, strides=[2, 2], name='conv10_2')  # 3
        pool6 = tf.nn.avg_pool(conv10_2, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")  # 1

    with tf.name_scope(name="textbox_layer") as scope:
        layers = [conv4_3, conv7, conv8_2, conv9_2, conv10_2, pool6]
        logits = []
        localisations = []
        for i, layer in enumerate(layers):
            with tf.variable_scope('{}_box'.format(i)):
                norm = 20 if i == 0 else -1
                sco_pred, loc_pred = _text_multibox_layer(layer, mode == tf.estimator.ModeKeys.TRAIN, is_last=(i == len(layers)),
                                            normalization=norm)
                logits.append(sco_pred)
                localisations.append(loc_pred)
    if mode != tf.estimator.ModeKeys.PREDICT:
        total_loss = _ssd_losses(logits, localisations, glocalisations, gscores,
                                 match_threshold=params['match_threshold'],
                                 negative_ratio=params['negative_ratio'],
                                 alpha=params['loss_alpha'],
                                 label_smoothing=params['label_smoothing'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(params['learning_rate'])
            train = optimizer.minimize(total_loss, global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        predictions=None,
        training_hooks=None,
        export_outputs=None,
        train_op=train)


class TextBoxEstimator(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _box_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config)

        super(TextBoxEstimator, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )


def anchors(img_shape, dtype=np.float32):
    feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_ratios = [1, 2, 3, 5, 7, 10]
    scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.90]
    return textbox_achor_all_layers(img_shape, feat_shapes, anchor_ratios, scales, 0.5, dtype)


def textbox_achor_all_layers(img_shape,
                             layers_shape,
                             anchor_ratios,
                             scales,
                             offset=0.5,
                             dtype=np.float32):
    """
    Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
                                                 anchor_ratios,
                                                 scales[i],
                                                 offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


def textbox_anchor_one_layer(img_shape,
                             feat_size,
                             ratios,
                             scale,
                             offset=0.5,
                             dtype=np.float32):
    # Follow the papers scheme
    # 12 ahchor boxes with out sk' = sqrt(sk * sk+1)
    y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]] + 0.5
    y = y.astype(dtype) / feat_size[0]
    x = x.astype(dtype) / feat_size[1]
    x_offset = x
    y_offset = y + offset
    x_out = np.stack((x, x_offset), -1)
    y_out = np.stack((y, y_offset), -1)
    y_out = np.expand_dims(y_out, axis=-1)
    x_out = np.expand_dims(x_out, axis=-1)

    #
    num_anchors = 6
    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)
    for i, r in enumerate(ratios):
        h[i] = scale / math.sqrt(r) / feat_size[0]
        w[i] = scale * math.sqrt(r) / feat_size[1]
    return y_out, x_out, h, w


def bboxes_encode(bboxes, anchors,
                    scope='text_bboxes_encode'):
    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    return util.textbox_common.tf_text_bboxes_encode(
            bboxes, anchors,
            matching_threshold=0.1,
            prior_scaling=prior_scaling,
            scope=scope)

def bboxes_encode_0(bboxes, anchors,
                  scope='text_bboxes_encode'):
    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    r1 = []
    r2 = []
    for s in tf.split(bboxes, bboxes.shape[0]):
        v1, v2 = util.textbox_common.tf_text_bboxes_encode(
            s, anchors,
            matching_threshold=0.1,
            prior_scaling=prior_scaling,
            scope=scope)
        r1.append(v1)
        r2.append(v2)
    return tf.concat(r1, 0), tf.concat(r2, 0)


def fake_fn(params, is_training):
    text_shape = (300, 300)
    text_anchors = anchors(text_shape)
    def _data():
        ds = tf.data.Dataset.range(params['batch_size']*10)
        def _fake(_):
            i = np.random.uniform(low=0.0, high=1.0, size=(300, 300, 3))
            b = [np.array([i*20+2,i*20+2,i*20+5,i*20+50],dtype=np.float32) for i in range(1)]
            b = np.stack(b)
            b /= 300
            l = np.ones([1, 1], dtype=np.int32)
            return i, b, l

        ds = ds.map(_fake)

        def _proccess(i, b, l):
            i, _, b = ssd_vgg_preprocessing.preprocess_for_train(i, l, b, text_shape)
            glocalisations, gscores = bboxes_encode(b, text_anchors)
            logging.info('gscores: {}'.format(gscores))
            return {'image': i,'glocalisations': tuple(glocalisations), 'gscores': tuple(gscores)},0

        ds = ds.map(_proccess)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        return ds

    return _data
