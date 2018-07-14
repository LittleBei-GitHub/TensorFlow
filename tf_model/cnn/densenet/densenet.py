# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import config

FLAGS = tf.app.flags.FLAGS
weights_decay = FLAGS.weights_decay


def weight_variable_truncated_norm(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def weight_variable_constant(shape):
    return tf.Variable(tf.constant(0.02, shape=shape))


def weight_variable_with_decay(shape):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    # 正则化损失
    weight_decay = tf.multiply(tf.nn.l2_loss(var), weights_decay, name='weight_loss')
    tf.add_to_collection('regularzation_loss', weight_decay)
    return var


def batch_norm(input_tensor, if_training):
    with tf.variable_scope('batch_normalization'):
        depth = int(input_tensor.get_shape()[-1])
        beta = tf.Variable(tf.constant(0.0, shape=[depth]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input_tensor, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(if_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed_tensor = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, 1e-3)
    return normed_tensor


def composite_function(input, growth_rate, if_training, name='composite'):
    with tf.name_scope(name):
        input_depth = int(input.get_shape()[-1])
        weights = weight_variable_with_decay([3, 3, input_depth, growth_rate])
        output = batch_norm(input, if_training)
        output = tf.nn.relu(output)
        output = tf.nn.conv2d(input=output,
                              filter=weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              data_format='NHWC',
                              name='composite_3x3_s1')
        return output


def transition_layer(input, theta, if_training, name='transition'):
    with tf.name_scope(name):
        input_depth = int(input.get_shape()[-1])
        weights = weight_variable_with_decay([1, 1, input_depth, int(theta * input_depth)])
        output = batch_norm(input, if_training)
        output = tf.nn.conv2d(input=output, filter=weights, strides=[1, 1, 1, 1], padding='SAME',
                                       data_format='NHWC', name='transition_1x1_s1')
        output = tf.nn.avg_pool(value=output,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                data_format='NHWC',
                                name='avg_pool_2x2')
        return output


def bottleneck_layer(input, growth_rate, if_training, name='bottleneck'):
    with tf.name_scope(name):
        input_depth = int(input.get_shape()[-1])
        # NOTE: output_tensor should be 4k, 4 times of the growth_rate
        weights = weight_variable_with_decay([1, 1, input_depth, 4 * growth_rate])
        output = batch_norm(input, if_training)
        output = tf.nn.relu(output)
        output = tf.nn.conv2d(input=output,
                              filter=weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              data_format='NHWC',
                              name='bottleneck_1x1_s1')
        # NOTE: output/input_tensor_depth is different from input
        output = composite_function(output, growth_rate, if_training)
        return output


def dense_block(input, growth_rate, if_training):
    input_depth = int(input.get_shape()[-1])
    with tf.name_scope("bottleneck_0") as scope:
        # NOTE:64 is 2k, here k = 32, 128 is 4k, output is k = 32
        bottleneck_output_0 = bottleneck_layer(input, growth_rate, if_training)
        bottleneck_input_0 = tf.concat(values=[input, bottleneck_output_0], axis=3, name='stack0')  # 96

    with tf.name_scope("bottleneck_1") as scope:
        bottlenect_output_1 = bottleneck_layer(bottleneck_input_0, growth_rate, if_training)  # NOTE:96 = 64 + 32
        bottleneck_input_1 = tf.concat(values=[bottleneck_input_0, bottlenect_output_1], axis=3, name='stack1')  # 128

    with tf.name_scope("bottleneck_2") as scope:
        bottlenect_output_2 = bottleneck_layer(bottleneck_input_1, growth_rate, if_training)  # NOTE:128 = 96 + 32
        bottleneck_input_2 = tf.concat(values=[bottleneck_input_1, bottlenect_output_2], axis=3, name='stack2')  # 160

    with tf.name_scope("bottleneck_3") as scope:
        bottlenect_output_3 = bottleneck_layer(bottleneck_input_2, growth_rate, if_training)  # NOTE:160 = 128 + 32
        bottleneck_input_3 = tf.concat(values=[bottleneck_input_2, bottlenect_output_3], axis=3, name='stack3')  # 192

    with tf.name_scope("bottleneck_4") as scope:
        bottlenect_output_4 = bottleneck_layer(bottleneck_input_3, growth_rate, if_training)  # NOTE:192 = 160 + 32
        bottleneck_input_4 = tf.concat(values=[bottleneck_input_3, bottlenect_output_4], axis=3, name='stack4')  # 224

    with tf.name_scope("bottleneck_5") as scope:
        bottlenect_output_5 = bottleneck_layer(bottleneck_input_4, growth_rate, if_training)  # NOTE:192 = 160 + 32
        output_tensor = tf.concat(values=[bottleneck_input_4, bottlenect_output_5], axis=3, name='stack5')

    return output_tensor


def densenet_inference(image_batch, if_training, dropout_prob):
    with tf.name_scope('DenseNet-BC-121'):
        image_batch = tf.reshape(image_batch, [-1, 224, 224, 3])

        with tf.name_scope('conv2d_7x7_s2') as scope:
            conv_weights = weight_variable_with_decay([7, 7, 3, 64])
            output_tensor = tf.nn.conv2d(input=image_batch, filter=conv_weights, strides=[1, 2, 2, 1],
                                          padding='SAME', data_format='NHWC', name='conv2d_7x7_s2')
            output_tensor = tf.nn.max_pool(value=output_tensor, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', data_format='NHWC', name='max_pool_3x3_s2')
            output_tensor = tf.nn.relu(output_tensor)

        with tf.name_scope('dense_block_0') as scope:
            output_tensor = dense_block(output_tensor, 32, if_training)

        with tf.name_scope('transition_layer_0') as scope:
            output_tensor = transition_layer(output_tensor, 0.5, if_training)

        with tf.name_scope('dense_block_1') as scope:
            output_tensor = dense_block(output_tensor, 32, if_training)

        with tf.name_scope('transition_layer_1') as scope:
            output_tensor = transition_layer(output_tensor, 0.5, if_training)

        with tf.name_scope('dense_block_2') as scope:
            output_tensor = dense_block(output_tensor, 32, if_training)

        with tf.name_scope('transition_layer_2') as scope:
            output_tensor = transition_layer(output_tensor, 0.5, if_training)

        with tf.name_scope('dense_block_3') as scope:
            output_tensor = dense_block(output_tensor, 32, if_training)

        with tf.name_scope('avg_pool_7x7') as scope:
            output_tensor = tf.nn.avg_pool(value=output_tensor, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1],
                                            padding='SAME', data_format='NHWC', name='avg_pool_7x7')

        with tf.name_scope('fc'):
            W_fc0 = weight_variable_with_decay([368, 128])
            b_fc0 = weight_variable_constant([128])
            output_tensor = tf.reshape(output_tensor, [-1, 368])
            output_tensor = tf.nn.relu(tf.matmul(output_tensor, W_fc0) + b_fc0)

            output_tensor = tf.nn.dropout(output_tensor, dropout_prob)

            W_fc1 = weight_variable_with_decay([128, 5])
            b_fc1 = weight_variable_constant([5])
            output_tensor = tf.nn.relu(tf.matmul(output_tensor, W_fc1) + b_fc1)

    return output_tensor