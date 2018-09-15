# coding=utf-8

## version1.0
## TensorFlow实现双隐层自编码器-计算图并运行

import tensorflow as tf


# 控制训练过程的参数
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# 网络模型参数
n_hidden1_units = 256  # 编码器第一隐藏层神经元数量（让编码器和解码器都有同样规模的隐藏层）
n_hidden2_units = 128  # 编码器第二隐藏层神经元数量（让编码器和解码器都有同样规模的隐藏层）
n_input_units = 784  # 输入层神经元数量MNIST data input(img shape :28*28)
n_output_units = n_input_units  # 解码器输出晨神经元数量必须等于输入数据的units数量


# 根据输入输出节点数量返回初始化好的指定名称的权重Variable
def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)


# 根据输出节点数量返回初始化好的指定名称的偏置Variable
def BiasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out]), dtype=tf.float32, name=name_str)


# 构建编码器
def Encoder(x_origin, activate_func=tf.nn.sigmoid):
    # 编码器第一隐藏层
    with tf.name_scope('Layer1'):
        weights = WeightsVariable(n_input_units, n_hidden1_units, 'weights')
        biases = BiasesVariable(n_hidden1_units, 'biases')
        x_code1 = activate_func(tf.add(tf.matmul(x_origin, weights), biases))

    # 编码器第二隐藏层
    with tf.name_scope('Layer2'):
        weights = WeightsVariable(n_hidden1_units, n_hidden2_units, 'weights')
        biases = BiasesVariable(n_hidden2_units, 'biases')
        x_code2 = activate_func(tf.add(tf.matmul(x_code1, weights), biases))
    return x_code2


# 构建解码器
def Decoder(x_code, activate_func):
    # 解码器第一隐藏层
    with tf.name_scope('Layer1'):
        weights = WeightsVariable(n_hidden2_units, n_hidden1_units, 'weights')
        biases = BiasesVariable(n_hidden1_units, 'biases')
        x_decode1 = activate_func(tf.add(tf.matmul(x_code, weights), biases))

    # 解码器第二隐藏层
    with tf.name_scope('Layer2'):
        weights = WeightsVariable(n_hidden1_units, n_output_units, 'weights')
        biases = BiasesVariable(n_output_units, 'biases')
        x_decode2 = activate_func(tf.add(tf.matmul(x_decode1, weights), biases))
    return x_decode2


# 调用上面写的函数构造计算图
with tf.Graph().as_default():
    # 计算图输入
    with tf.name_scope('X_origin'):
        X_Origin = tf.placeholder(tf.float32, [None, n_input_units])

    # 构建编码器模型
    with tf.name_scope('Encoder'):
        X_code = Encoder(X_Origin, activate_func=tf.nn.sigmoid)

    # 构建解码器模型
    with tf.name_scope('Decoder'):
        X_decode = Decoder(X_code, activate_func=tf.nn.sigmoid)

    # 定义损失节点：重构数据与原始数据的误差平方和损失
    with tf.name_scope('Loss'):
        Loss = tf.reduce_mean(tf.pow(X_Origin - X_decode, 2))

    # 定义优化器，训练节点
    with tf.name_scope('Train'):
        Optimizer = tf.train.RMSPropOptimizer(learning_rate)
        Train = Optimizer.minimize(Loss)

    # 为所有变量添加初始化节点
    Init = tf.global_variables_initializer()

    print('把计算图写入事件文件，在Tesorboard里面查看')
    summary_write = tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
    summary_write.flush()