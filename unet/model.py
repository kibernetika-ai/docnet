import tensorflow as tf
import numpy as np
from unet.unet import unet
from kibernetika.rpt import MlBoardReporter
import glob


def data_fn(params, training):
    data_set = params['data_set']
    datasets_files = []
    for tf_file in glob.iglob(data_set+'/*.record'):
        datasets_files.append(tf_file)
    resolution = params['resolution']
    fresolution = float(resolution)
    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)
        def _draw(xmin,xmax,ymin,ymax):
            img = np.zeros((resolution,resolution,1),np.float32)
            for i in range(len(xmin)):
                y1 = max(0,ymin[i])
                y2 = min(resolution,ymax[i])
                x1 = max(0,xmin[i])
                x2 = min(resolution,xmax[i])
                img[y1:y2,x1:x2,0]=1
            return img
        def _parser(example):
            features = {
                'image/encoded':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/height':
                    tf.FixedLenFeature([1], tf.int64),
                'image/width':
                    tf.FixedLenFeature([1], tf.int64),
                'image/object/bbox/xmin':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmax':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymin':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymax':
                    tf.VarLenFeature(tf.float32)
            }
            res = tf.parse_single_example(example, features)
            img = tf.image.decode_png(res['image/encoded'], channels=3)
            original_w = tf.cast(res['image/width'][0], tf.int32)
            original_h = tf.cast(res['image/height'][0], tf.int32)
            img = tf.reshape(img, [1,original_h, original_w, 3])
            img = tf.image.resize_images(img,[resolution,resolution])[0]
            xmin = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmin'])*fresolution,tf.int32)
            xmax = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmax'])*fresolution,tf.int32)
            ymin = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymin'])*fresolution,tf.int32)
            ymax = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymax'])*fresolution,tf.int32)
            mask = tf.py_func(_draw, [xmin,xmax,ymin,ymax], tf.float32)
            return img,mask
        ds = ds.map(_parser)
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        if training:
            ds = ds.repeat(params['num_epochs']).prefetch(params['batch_size'])
        return ds
    return _input_fn


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['image']
    else:
        features = tf.reshape(features, [params['batch_size'], params['resolution'], params['resolution'], 3])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    out_chans = 1
    logits = unet(features, out_chans, params['num_chans'], params['drop_prob'], params['num_pools'], training=training)
    mask = tf.sigmoid(logits)
    loss = None
    train_op = None
    hooks = []
    export_outputs = None
    eval_hooks = []
    chief_hooks = []
    metrics = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
        original = features * labels
        predicted = features * mask
        loss = tf.losses.absolute_difference(labels, mask)
        mse = tf.losses.mean_squared_error(labels, mask)
        nmse = tf.norm(labels - mask) ** 2 / tf.norm(labels) ** 2

        global_step = tf.train.get_or_create_global_step()
        if training:
            tf.summary.scalar('mse', mse)
            tf.summary.scalar('nmse', nmse)
            board_hook = MlBoardReporter({
                "_step": global_step,
                "_train_loss": loss,
                '_train_mse': mse,
                '_train_nmse': nmse}, every_steps=params['save_summary_steps'])
            chief_hooks = [board_hook]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if params['optimizer'] == 'AdamOptimizer':
                    opt = tf.train.AdamOptimizer(float(params['lr']))
                else:
                    opt = tf.train.RMSPropOptimizer(float(params['lr']), params['weight_decay'])
                train_op = opt.minimize(loss, global_step=global_step)

        tf.summary.image('Src', features, 3)
        tf.summary.image('Reconstruction', predicted, 3)
        tf.summary.image('Original', original, 3)
        hooks = []
        if not training:
            metrics['mse'] = tf.metrics.mean(mse)
            metrics['nmse'] = tf.metrics.mean(nmse)
    else:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                mask)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=mask,
        training_chief_hooks=chief_hooks,
        loss=loss,
        training_hooks=hooks,
        export_outputs=export_outputs,
        evaluation_hooks=eval_hooks,
        train_op=train_op)




class BoxUnet(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _unet_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(BoxUnet, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
