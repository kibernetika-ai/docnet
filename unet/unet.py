import tensorflow as tf
import logging

def conv_block(input, out_chans, drop_prob, name, pooling, training):
    with tf.variable_scope("layer_{}".format(name)):
        out = input
        for j in range(2):
            out = tf.layers.conv2d(out, out_chans, kernel_size=3, padding='same', name="conv_{}".format(j + 1))
            out = tf.layers.batch_normalization(out, training=training, name="bn_{}".format(j + 1))
            out = tf.nn.relu(out, name="relu_{}".format(j + 1))
            if training:
                out = tf.layers.dropout(out, drop_prob, name="dropout_{}".format(j + 1))

        if not pooling:
            return out

        pool = tf.nn.max_pool(out,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool')
        return out, pool


def unpool(pool, ind, ksize=[1, 2, 2, 1], name=None):
    with tf.variable_scope(name) as scope:
        mask = tf.cast(ind, tf.int32)
        input_shape = tf.shape(pool, out_type=tf.int32)
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3]) % output_shape[1]
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range
        updates_size = tf.size(pool)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(pool, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

def _fake_conv2d_transpose(output,f,name):
    l = output.get_shape().as_list()
    output = tf.image.resize_bilinear(output,(l[1]*2,l[2]*2),name='{}_resize'.format(name))
    return tf.layers.conv2d(output,f,3, strides=(1, 1), padding='same',name='{}_conv'.format(name))

def upnet(name,output,num_pool_layers,down_sample_layers,ch,drop_prob,training):
    for i in range(num_pool_layers):
        down = down_sample_layers[len(down_sample_layers)-i-1]
        _,w,h,f = down.shape
        #output = tf.layers.conv2d_transpose(output,filters=f,kernel_size=[3, 3],strides=[2, 2],padding='SAME',
        #                                    activation=None,
        #                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        #                                    name='unpool{}_{}'.format(name,i + 1))
        output = _fake_conv2d_transpose(output,f,'unpool{}_{}'.format(name,i + 1))
        output = tf.concat([output, down], 3)
        logging.info('Up{}_{} - {}'.format(name,i+1,output.shape))
        if i < (num_pool_layers-1):
            ch //= 2
        output = conv_block(output, ch, drop_prob, 'up{}_{}'.format(name,i + 1), False, training)
    return output

def unet(inputs, out_chans, chans, drop_prob, num_pool_layers,training=True,up_type='join'):
    output, pull = conv_block(inputs, chans, drop_prob, 'down_1', True, training)
    logging.info('Down_1 - {}'.format(output.shape))
    down_sample_layers = [output]
    ch = chans
    for i in range(num_pool_layers - 1):
        ch *= 2
        output, pull = conv_block(pull, ch, drop_prob, 'down_{}'.format(i + 2), True, training)
        logging.info('Down_{} - {}'.format(i+2,output.shape))
        down_sample_layers += [output]
    i+=1

    final_down = conv_block(pull, ch, drop_prob, 'down_{}'.format(i + 2), False, training)
    logging.info('Down_{} - {}'.format(i+2,output.shape))

    if up_type=='seprate':
        outs = []
        for i,out_chan in enumerate(out_chans):
            name = '_{}'.format(i)
            output = upnet(name,final_down,num_pool_layers,down_sample_layers,ch,drop_prob,training)
            output = tf.layers.conv2d(output, ch, kernel_size=1, padding='same', name="conv{}_1".format(name))
            output = tf.layers.conv2d(output, out_chan, kernel_size=1, padding='same', name="conv{}_2".format(name))
            output = tf.layers.conv2d(output, out_chan, kernel_size=1, padding='same', name="final{}".format(name))
            outs.append(output)
        return outs
    else:
        output = upnet('',final_down,num_pool_layers,down_sample_layers,ch,drop_prob,training)
        output = tf.layers.conv2d(output, ch, kernel_size=1, padding='same', name="conv_1")
        output1 = tf.layers.conv2d(output, out_chans[0], kernel_size=1, padding='same', name="conv_2_mask")
        output2 = tf.layers.conv2d(output1, out_chans[0], kernel_size=1, padding='same', name="final_mask")
        output3 = tf.layers.conv2d(output, out_chans[1], kernel_size=1, padding='same', name="conv_2_links")
        output4 = tf.layers.conv2d(output3, out_chans[1], kernel_size=1, padding='same', name="final_links")
        return [output2,output4]