#coding=utf-8

## 实现简单卷积神经网络对MNIST数据集进行分类
## conv2d + activation + pool + fc
import csv
import tensorflow as tf
# import cifar10_input

# 设置算法超参数
learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
# n_classes = 10 # MNIST total classes (0-9 digits)

#数据集中输入图像的参数
image_size = 24
image_channel = 3
n_classes = 10 #CiFar10中类的数量

#根据指定的维数返回初始化好的指定名称的权重 Variable
def WeightsVariable(shape, name_str, stddev=0.1):
    # initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

#根据指定的维数返回初始化好的指定名称的偏置 Variable
def BiasesVariable(shape, name_str, init_value=0.00001):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

# 二维卷积层activation(conv2d+bias)的封装
def Conv2d(x, W, b, stride=1, padding='SAME',activation=tf.nn.relu,act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y

# 二维池化层pool的封装
def Pool2d(x, pool= tf.nn.max_pool, k=2, stride=2,padding='SAME'):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)

# 全连接层activate(wx+b)的封装
def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y

def Inference(image_holder):
    # 第一个卷积层activate(conv2d + biase)
    with tf.name_scope('Conv2d_1'):
        conv1_kernels_num = 64
        weights = WeightsVariable(shape=[5, 5, image_channel, conv1_kernels_num],
                                  name_str='weights',stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernels_num], name_str='biases',init_value=0.0)
        conv1_out = Conv2d(image_holder, weights, biases, stride=1, padding='SAME')

    # 第一个池化层(pool 2d)
    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2,padding='SAME')

    # 第二个卷积层activate(conv2d + biase)
    with tf.name_scope('Conv2d_2'):
        conv2_kernels_num = 64
        weights = WeightsVariable(shape=[5, 5, conv1_kernels_num, conv2_kernels_num],
                                  name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv2_kernels_num], name_str='biases', init_value=0.0)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')

    # 第二个池化层(pool 2d)
    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

    #将二维特征图变换为一维特征向量
    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool2_out, [batch_size,-1])   # -1表示height*width*channel
        feats_dim = features.get_shape()[1].value

    # 第一个全连接层(fully connected layer)
    with tf.name_scope('FC1_nonlinear'):
        fc1_units_num = 384
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num],
                                  name_str='weights',stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num], name_str='biases',init_value=0.1)
        fc1_out = FullyConnected(features, weights, biases, activate=tf.nn.relu, act_name='relu')

    # 第二个全连接层(fully connected layer)
    with tf.name_scope('FC2_nonlinear'):
        fc2_units_num = 192
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num],
                                  name_str='weights',stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num], name_str='biases',init_value=0.1)
        fc2_out = FullyConnected(fc1_out, weights, biases,activate=tf.nn.relu, act_name='relu')

    # 第三个全连接层(fully connected layer)
    with tf.name_scope('FC3_linear'):
        fc3_units_num = n_classes
        weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num],
                                  name_str='weights',stddev=1.0/fc2_units_num)
        biases = BiasesVariable(shape=[fc3_units_num], name_str='biases',init_value=0.0)
        logits = FullyConnected(fc2_out, weights, biases,activate=tf.identity, act_name='linear')
    return logits

#调用上面写的函数构造计算图
with tf.Graph().as_default():

    # 计算图输入
    with tf.name_scope('Inputs'):
        image_holder = tf.placeholder(tf.float32, [batch_size, image_size,image_size,image_channel], name='images')
        labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

    # 计算图前向推断过程
    with tf.name_scope('Inference'):
         logits = Inference(image_holder)

    # 定义损失层(loss layer)
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder,logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        total_loss = cross_entropy_mean

    # 定义优化训练层(train layer)
    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        trainer_op = optimizer.minimize(total_loss,global_step=global_step)

    # 定义模型评估层(evaluate layer)
    with tf.name_scope('Evaluate'):
        top_K_op = tf.nn.in_top_k(predictions=logits,targets=labels_holder,k=1)

    # 添加所有变量的初始化节点
    init_op = tf.global_variables_initializer()

    print('把计算图写入事件文件，在TensorBoard里面查看')
    graph_writer = tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
    graph_writer.close()