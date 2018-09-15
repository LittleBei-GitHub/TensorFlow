# coding=utf-8

"""Contains common code shared by all inception models.
Usage of arg scope:
  with slim.arg_scope(inception_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes,
                                                is_training=is_training)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import slim
import tensorflow as tf


def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu):
    """Defines the default arg scope for inception models.
       给inception models 定义默认的 arg scope
    Args:
      weight_decay: The weight decay to use for regularizing the model.
                    权重衰减用于模型正则化
      use_batch_norm: "If `True`, batch_norm is applied after each convolution.
                      如果true，在每一次卷积操作之后应用批量正则化
      batch_norm_decay: Decay for batch norm moving average.
                        衰减用于批量正则化中的moving average
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero in batch norm.
                          平滑操作，避免除数为0
      activation_fn: Activation function for conv2d.
                     卷积操作的激活函数
    Returns:
      An `arg_scope` to use for the inception models.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # use fused batch norm if possible.
        'fused': None,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers.
    # 给卷积层和全连接层的权重设置权重衰减
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params) as sc:
            return sc
