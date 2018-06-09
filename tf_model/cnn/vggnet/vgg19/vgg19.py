# coding=utf-8

import tensorflow as tf

CLASS_NUM = 10
DROPOUT_KEEP_PROB = 0.7

def weights_variable(shape, stddev=0.1, name='weights'):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, name=name)


def biases_variable(shape, name='biases'):
    initial = tf.constant(vaule=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d(input, weigths, name='conv'):
    return tf.nn.conv2d(input=input,
                        filter=weigths,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name=name)


def conv2d(input, filter_height, filter_width, num_channels, num_filters, name):
    with tf.name_scope(name):
        weigths = weights_variable(shape=[filter_height, filter_width, num_channels, num_filters],
                                   stddev=0.1)
        biases = biases_variable(shape=[num_filters])
        conv = conv2d(input=input, weigths=weigths)
        conv_out = tf.nn.relu(features=tf.nn.bias_add(value=conv, bias=biases), name='relu')
        return conv_out


def pool_max(input, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool')


def fc(input, num_in, num_out, name, active=True):
    with tf.name_scope(name):
        weights = weights_variable(shape=[num_in, num_out])
        biases = biases_variable(shape=[num_out])
        fc = tf.matmul(input, weights) + biases
        if active:
            fc_out = tf.nn.relu(features=fc, name='relu')
            return fc_out
        else:
            return fc


def dropout(input, keep_prob, name):
    with tf.name_scope(name):
        return tf.nn.dropout(x=input, keep_prob=keep_prob, name='dropout')


def inference(inputs, train=True):
    with tf.name_scope('conv1'):
        conv1_1 = conv2d(input=inputs, filter_height=3, filter_width=3, num_channels=3, num_filters=64,
                         name='conv1_1')
        conv1_2 = conv2d(input=conv1_1, filter_height=3, filter_width=3, num_channels=64, num_filters=64,
                         name='conv1_2')
        pool1 = pool_max(input=conv1_2, name='pool1')

    with tf.name_scope('conv2'):
        conv2_1 = conv2d(input=pool1, filter_height=3, filter_width=3, num_channels=64, num_filters=128,
                         name='conv2_1')
        conv2_2 = conv2d(input=conv2_1, filter_height=3, filter_width=3, num_channels=128, num_filters=128,
                         name='conv2_2')
        pool2 = pool_max(input=conv2_2, name='pool2')

    with tf.name_scope('conv3'):
        conv3_1 = conv2d(input=pool2, filter_height=3, filter_width=3, num_channels=128, num_filters=256,
                         name='conv3_1')
        conv3_2 = conv2d(input=conv3_1, filter_height=3, filter_width=3, num_channels=256, num_filters=256,
                         name='conv3_2')
        conv3_3 = conv2d(input=conv3_2, filter_height=3, filter_width=3, num_channels=256, num_filters=256,
                         name='conv3_3')
        conv3_4 = conv2d(input=conv3_3, filter_height=3, filter_width=3, num_channels=256, num_filters=256,
                         name='conv3_4')
        pool3 = pool_max(conv3_4, name='pool3')

    with tf.name_scope('conv4'):
        conv4_1 = conv2d(input=pool3, filter_height=3, filter_width=3, num_channels=256, num_filters=512,
                         name='conv4_1')
        conv4_2 = conv2d(input=conv4_1, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv4_2')
        conv4_3 = conv2d(input=conv4_2, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv4_3')
        conv4_4 = conv2d(input=conv4_3, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv4_4')
        pool4 = pool_max(conv4_4, name='pool4')

    with tf.name_scope('conv5'):
        conv5_1 = conv2d(input=pool4, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_1')
        conv5_2 = conv2d(input=conv5_1, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_2')
        conv5_3 = conv2d(input=conv5_2, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_3')
        conv5_4 = conv2d(input=conv5_3, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_4')
        pool5 = pool_max(conv5_4, name='pool5')

    with tf.name_scope('fc6') as scope:
        pool5_shape = pool5.get_shape().as_list()
        nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        pool5_flat = tf.reshape(tensor=pool5, shape=[-1, nodes], name='flat')
        fc6 = fc(input=pool5_flat, num_in=nodes, num_out=4096, name=scope)
        if train:
            fc6_dropout = dropout(input=fc6, keep_prob=DROPOUT_KEEP_PROB, name=scope)
        else:
            fc6_dropout = dropout(input=fc6, keep_prob=1.0, name=scope)

    with tf.name_scope('fc7') as scope:
        fc7 = fc(input=fc6_dropout, num_in=4096, num_out=4096, name=scope)
        if train:
            fc7_dropout = dropout(input=fc7, keep_prob=DROPOUT_KEEP_PROB, name=scope)
        else:
            fc7_dropout = dropout(input=fc7, keep_prob=1.0, name=scope)

    with tf.name_scope('fc8') as scope:
        fc8 = fc(input=fc7_dropout, num_in=4096, num_out=CLASS_NUM, name=scope)

    return fc8


if __name__ == '__main__':
    with tf.name_scope('Inference'):
        y_hat = inference()

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=None, logits=y_hat)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('Train'):
        global_step = tf.Variable(initial_value=0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer.minimize(loss=loss, global_step=global_step)

    with tf.name_scope('Evaluate'):
        correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(None, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
