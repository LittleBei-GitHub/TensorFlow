# coding=utf-8

import tensorflow as tf

# 定义神经网络相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 6
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 5

# 第三层卷积层的尺寸和深度
CONV3_DEEP=120
CONV3_SIZE=5

# 全连接层的节点个数
FC_SIZE = 84

# 定义神经网络的前向传播过程。
# 这里添加了一个新的参数train，用于区别训练过程和测试过程。
# 在这个程序中将用到dropout方法，dropout可以进一步提升模型可靠性并防止过拟合，dropout过程只在训练时使用。
def inference(input_tensor, train, regularizer):
    ## 第一层卷积层
    # 和标准LeNet-5模型不大一样，这里定义卷积层的输入为28*28*1的原始MNIST图片像素。
    # 因为卷积层中使用了全0填充，所以输出为28*28*32的矩阵。
    with tf.name_scope('layer1_conv1'):

        conv1_weights=tf.Variable(
            initial_value=tf.truncated_normal(shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                                                    stddev=0.1,
                                                                    dtype=tf.float32),
            dtype=tf.float32,
            name='conv1_weights')
        conv1_biases=tf.Variable(
            initial_value=tf.constant(value=0.0, shape=[CONV1_DEEP]),
            dtype=tf.float32,
            name='conv1_biases')
        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1_out = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    ## 第二层池化层
    # 这里选用最大池化层，池化层过滤器的边长为2，使用全0填充且移动的步长为2。
    # 这一层的输入是上一层的输出，也就是28*28*32的矩阵。输出为14*14*32的矩阵。
    with tf.name_scope('layer2_pool'):
        pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ## 第三层卷积层
    # 这一层的输入为14*14*32的矩阵，输出为14*14*64的矩阵。
    with tf.name_scope('layer3_conv2'):
        conv2_weights=tf.Variable(
            initial_value=tf.truncated_normal(shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                                                    stddev=0.1,
                                                                    dtype=tf.float32),
            dtype=tf.float32,
            name='conv2_weights')
        conv2_biases=tf.Variable(
            initial_value=tf.constant(value=0.0, shape=[CONV2_DEEP]),
            dtype=tf.float32,
            name='conv2_biases')
        # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2_out = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    ## 第四层池化层
    # 这一层和第二层的结构是一样的。这一层的输入为14*14*64的矩阵，输出为7*7*64的矩阵。
    with tf.name_scope('layer4_poo2'):
        pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ## 第五层卷积层
    with tf.name_scope('layer5_conv3'):
        conv3_weights = tf.Variable(
            initial_value=tf.truncated_normal(shape=[CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                              stddev=0.1,
                                              dtype=tf.float32),
            dtype=tf.float32,
            name='conv3_weights')
        conv3_biases = tf.Variable(
            initial_value=tf.constant(value=0.0, shape=[CONV3_DEEP]),
            dtype=tf.float32,
            name='conv3_biases')
        # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且使用全0填充
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3_out = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))


    # conv3_out.get_shape函数可以得到第四层输出矩阵的维度而不需要手工计算。
    # 注意因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数。
    conv3_shape = conv3_out.get_shape().as_list()
    # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长度及深度的乘积。
    # 注意这里pool_shape[0]为一个batch中样本的个数。
    nodes = conv3_shape[1] * conv3_shape[2] * conv3_shape[3]
    # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
    conv3_flat = tf.reshape(conv3_out, [conv3_shape[0], nodes])

    ## 第六层全连接层


    # 这一层的输入是拉直之后的一组向量，向量长度为7*7*64=3136，输出是一组长度为512的向量。
    # 这一层和之前在第5章中介绍的基本一致，唯一的区别是引入了dropout的概念。
    # dropout在训练时会随机将部分节点的输出改为0。
    # dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好。
    # dropout一般只在全连接层而不是卷积层或者池化层使用。
    with tf.variable_scope('layer6_fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer = tf.truncated_normal_initializer(stddev = 0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer = tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播过程。
    # 这一层的输入是一组长度为512的向量，输出是一组长度为10的向量。
    # 这一层的输出通过Softmax之后就得到了最后的分类结果。
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer = tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    # 返回第六层的输出
    return logit