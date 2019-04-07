import tensorflow as tf
from unet.unet import unet
from kibernetika.rpt import MlBoardReporter


def _flat_pixel_cls_values(values):
    shape = values.shape.as_list()
    values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
    return values


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['image']
    else:
        features = tf.reshape(features, [params['batch_size'], params['resolution'], params['resolution'], 3])
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    pixel_cls_logits, pixel_link_logits = unet(features, [2, 16], params['num_chans'], params['drop_prob'],
                                               params['num_pools'], training=training)

    pixel_cls_scores = tf.nn.softmax(pixel_cls_logits)
    pixel_cls_logits_flatten = _flat_pixel_cls_values(pixel_cls_logits)
    pixel_cls_scores_flatten = _flat_pixel_cls_values(pixel_cls_scores)
    shape = tf.shape(pixel_link_logits)
    pixel_link_logits = tf.reshape(pixel_link_logits, [shape[0], shape[1], shape[2], 8, 2])

    pixel_link_scores = tf.nn.softmax(pixel_link_logits)

    pixel_pos_scores = pixel_cls_scores[:, :, :, 1]
    link_pos_scores = pixel_link_scores[:, :, :, :, 1]

    loss = None
    train_op = None
    hooks = []
    export_outputs = None
    eval_hooks = []
    chief_hooks = []
    metrics = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
        pixel_cls_loss,pixel_pos_link_loss,pixel_neg_link_loss = build_loss(params,
                   pixel_cls_logits_flatten, pixel_cls_scores_flatten, pixel_link_logits,
                   labels['pixel_cls_label'], labels['pixel_cls_weight'],
                   labels['pixel_link_label'], labels['pixel_link_weight'])
        loss = tf.get_collection(tf.GraphKeys.LOSSES)
        loss = tf.add_n(loss)
        original = features * tf.expand_dims(tf.cast(labels['pixel_cls_label'],tf.float32),-1)
        predicted = features * tf.expand_dims(pixel_pos_scores,-1)
        global_step = tf.train.get_or_create_global_step()
        if training:
            board_hook = MlBoardReporter({
                "_step": global_step,
                "_train_loss": loss}, every_steps=params['save_summary_steps'])
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
            metrics['pixel_cls_loss'] = tf.metrics.mean(pixel_cls_loss)
            metrics['pixel_pos_link_loss'] = tf.metrics.mean(pixel_pos_link_loss)
            metrics['pixel_neg_link_loss'] = tf.metrics.mean(pixel_neg_link_loss)
    else:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                {'pixel_pos_scores':pixel_pos_scores,'link_pos_scores':link_pos_scores})}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions= {'pixel_pos_scores':pixel_pos_scores,'link_pos_scores':link_pos_scores},
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


def build_loss(params,
               pixel_cls_logits_flatten, pixel_cls_scores_flatten, pixel_link_logits,
               pixel_cls_labels, pixel_cls_weights,
               pixel_link_labels, pixel_link_weights):
    batch_size = params['batch_size']
    ignore_label = -1
    background_label = 0
    text_label = 1
    pixel_link_neg_loss_weight_lambda = 1.0
    pixel_cls_loss_weight_lambda = 2.0
    pixel_link_loss_weight = 1.0

    def OHNM_single_image(scores, n_pos, neg_mask):
        def has_pos():
            return n_pos * 3

        def no_pos():
            return tf.constant(10000, dtype=tf.int32)

        n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
        max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))

        n_neg = tf.minimum(n_neg, max_neg_entries)
        n_neg = tf.cast(n_neg, tf.int32)

        def has_neg():
            neg_conf = tf.boolean_mask(scores, neg_mask)
            vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
            threshold = vals[-1]  # a negtive value
            selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
            return selected_neg_mask

        def no_neg():
            selected_neg_mask = tf.zeros_like(neg_mask)
            return selected_neg_mask

        selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
        return tf.cast(selected_neg_mask, tf.int32)

    def OHNM_batch(neg_conf, pos_mask, neg_mask):
        selected_neg_mask = []
        for image_idx in range(batch_size):
            image_neg_conf = neg_conf[image_idx, :]
            image_neg_mask = neg_mask[image_idx, :]
            image_pos_mask = pos_mask[image_idx, :]
            n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
            selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

        selected_neg_mask = tf.stack(selected_neg_mask)
        return selected_neg_mask

    # OHNM on pixel classification task
    pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [batch_size, -1])
    pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [batch_size, -1])

    pos_mask = tf.equal(pixel_cls_labels_flatten, text_label)
    neg_mask = tf.equal(pixel_cls_labels_flatten, background_label)

    n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))

    with tf.name_scope('pixel_cls_loss'):
        def no_pos():
            return tf.constant(.0);

        def has_pos():
            pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pixel_cls_logits_flatten,
                labels=tf.cast(pos_mask, dtype=tf.int32))

            pixel_neg_scores = pixel_cls_scores_flatten[:, :, 0]
            selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)

            pixel_cls_weights = pos_pixel_weights_flatten + \
                                tf.cast(selected_neg_pixel_mask, tf.float32)
            n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
            loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
            return loss

        #             pixel_cls_loss = tf.cond(n_pos > 0, has_pos, no_pos)
        pixel_cls_loss = has_pos()
        tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * pixel_cls_loss_weight_lambda)

    with tf.name_scope('pixel_link_loss'):
        def no_pos():
            return tf.constant(.0), tf.constant(.0);

        def has_pos():
            pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pixel_link_logits,
                labels=pixel_link_labels)

            def get_loss(label):
                link_mask = tf.equal(pixel_link_labels, label)
                link_weights = pixel_link_weights * tf.cast(link_mask, tf.float32)
                n_links = tf.reduce_sum(link_weights)
                loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
                return loss

            neg_loss = get_loss(0)
            pos_loss = get_loss(1)
            return neg_loss, pos_loss

        pixel_neg_link_loss, pixel_pos_link_loss = \
            tf.cond(n_pos > 0, has_pos, no_pos)

        pixel_link_loss = pixel_pos_link_loss + \
                          pixel_neg_link_loss * pixel_link_neg_loss_weight_lambda

        tf.add_to_collection(tf.GraphKeys.LOSSES,
                             pixel_link_loss_weight * pixel_link_loss)

    tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
    tf.summary.scalar('pixel_pos_link_loss', pixel_pos_link_loss)
    tf.summary.scalar('pixel_neg_link_loss', pixel_neg_link_loss)
    return pixel_cls_loss,pixel_pos_link_loss,pixel_neg_link_loss
