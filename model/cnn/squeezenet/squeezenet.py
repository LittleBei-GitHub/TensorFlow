# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def weights_variable(shape, stddev=0.1, name='weights'):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, name=name)


def biases_variable(shape, name='biases'):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name)


def conv2d(input, filter_height, filter_width, num_channels, num_filters, strides=[1, 1, 1, 1], name='conv2d'):
    with tf.name_scope(name):
        weigths = weights_variable(shape=[filter_height, filter_width, num_channels, num_filters],
                                   stddev=0.1)
        biases = biases_variable(shape=[num_filters])
        conv = tf.nn.conv2d(input=input,
                            filter=weigths,
                            strides=strides,
                            padding='SAME',
                            name=name)
        conv_out = tf.nn.relu(features=tf.nn.bias_add(value=conv, bias=biases), name='relu')
        return conv_out
def squeeze(input, num_channels, num_filters, name='squeeze'):
    with tf.name_scope(name) as scope:
        return conv2d(input, 1, 1, num_channels, num_filters, name=scope)

def expand(input, num_channels, num_filters_1, num_filters_3, name='expand'):
    with tf.name_scope(name) as scope:
        expand_1 = conv2d(input, 1, 1, num_channels, num_filters_1, name=scope+'_1')
        expand_3 = conv2d(input, 3, 3, num_channels, num_filters_3, name=scope+'_3')
        out = tf.concat(values=[expand_1, expand_3], axis=3, name=scope+'_concat')
        return out

def fire_module(input, num_channels, s_num_filters, e_num_filters_1, e_num_filters_3, name):
    with tf.name_scope(name) as scope:
        net = squeeze(input, num_channels, s_num_filters)
        net = expand(net, s_num_filters, e_num_filters_1, e_num_filters_3)
        return net

def squeeze_net(input, classes):
    weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 1, 96])),
               'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes]))}

    biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
              'conv10': tf.Variable(tf.truncated_normal([classes]))}

    net = tf.nn.conv2d(input, weights['conv1'], strides=[1, 2, 2, 1], padding='SAME', name='conv1')
    net = tf.nn.bias_add(net, biases['conv1'])

    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')

    net = fire_module(net, 96, 16, 64, 64, name='fire_module_2')
    net = fire_module(net, 128, 16, 64, 64, name='fire_module_3')
    net = fire_module(net, 128, 32, 128, 128, name='fire_module_4')

    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    net = fire_module(net, 256, 32, 128, 128, name='fire_module_5')
    net = fire_module(net, 256, 48, 192, 192, name='fire_module_6')
    net = fire_module(net, 384, 48, 192, 192, name='fire_module_7')
    net = fire_module(net, 384, 64, 256, 256, name='fire_module_8')

    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')

    net = fire_module(net, 512, 64, 256, 256, name='fire_module_9')

    net = tf.nn.dropout(net, keep_prob=0.5, name='dropout_9')

    net = tf.nn.conv2d(net, weights['conv10'], strides=[1, 1, 1, 1], padding='SAME', name='conv10')
    net = tf.nn.bias_add(net, biases['conv10'])

    net = tf.nn.avg_pool(net, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')

    return net