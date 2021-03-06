# coding=utf-8

from tensorflow.contrib import slim
import tensorflow as tf
import inception_utils

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      scope='InceptionV1'):
    """Defines the Inception V1 base architecture.
       定义 inception v1 的基本结构
    This architecture is defined in:
    该结构被定义在下面的这篇论文中
      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      输入:一个张量，它的结构为[批量大小, 图像高, 图像宽, 通道数]
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
        'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
        'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
        'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
      scope: Optional variable_scope.
      作用域: 可选的variable_scope
    Returns:
      A dictionary from components of the network to the corresponding activation.
      返回一个字典，从网络的组件到相应的激活层
    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values.
    """
    end_points = {}
    with tf.variable_scope(scope, 'InceptionV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=trunc_normal(0.01)):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                # 第一组
                end_point = 'Conv2d_1a_7x7'
                net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
                # 第二组
                end_point = 'MaxPool_2a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Conv2d_2b_1x1'
                net = slim.conv2d(net, 64, [1, 1], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Conv2d_2c_3x3'
                net = slim.conv2d(net, 192, [3, 3], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
                # 第三组
                end_point = 'MaxPool_3a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_3b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_3c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                # 第四组
                end_point = 'MaxPool_4a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4e'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4f'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                # 第五组
                end_point = 'MaxPool_5a_2x2'
                net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_5b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_5c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1',
                 global_pool=False):
    """Defines the Inception V1 architecture.
       定义inception v1的结构
    This architecture is defined in:
      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
    The default image size used to train this network is 224x224.
    使用224x224的图像训练网络
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
             一个大小为[batch_size, height, width, channels]的张量
      num_classes: number of predicted classes. If 0 or None, the logits layer
                   预测类别的个数
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      is_training: whether is training or not.
                   是否需要训练
      dropout_keep_prob: the percentage of activation values that are retained.
                         dropout的比例
      prediction_fn: a function to get predictions out of logits.
                     获取输出值logits预测的函数
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
                       如果true，logits的形状为[B, C]，如果false，logits的形状为
          shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          [B, 1, 1, C]，其中B是批量大小，C是类别个数
      reuse: whether or not the network and its variables should be reused. To be
             是否重用网络及其变量
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
             可选的variable_scope
      global_pool: Optional boolean flag to control the avgpooling before the
                   可选的布尔标志控制平均池化在logits层之前。
        logits layer. If false or unset, pooling is done with a fixed window
                      如果false或unset，pooling被做使用一个固定大小的窗口
        that reduces default-sized inputs to 1x1, while larger inputs lead to
             将默认大小的输入缩减到1x1
        larger outputs. If true, any input size is pooled down to 1x1.
                        如果true，任何输入大小被池化成1x1
    Returns:
      net: a Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    # Final pooling and prediction
    with tf.variable_scope(scope, 'InceptionV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v1_base(inputs, scope=scope)
            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with a fixed kernel size.
                    net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
                    end_points['AvgPool_0a_7x7'] = net
                if not num_classes:
                    return net, end_points
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_0c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


inception_v1.default_image_size = 224

inception_v1_arg_scope = inception_utils.inception_arg_scope
