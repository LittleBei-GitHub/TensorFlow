#coding=utf-8

## version1.0
## 实现简单卷积神经网络对MNIST数据集进行分类
## conv2d + activation + pool + fc

import tensorflow as tf

#设置算法超参数
learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

#Network Parameters
n_input = 784 #MNIST data input(img shape:28*28)
n_class = 10 #MNIST total classes(0-9 digits)

#根据指定的维数返回初始化好的指定名称和权重 Variable
def WeightsVariable(shape,name_str,stddev=0.1):
    initial = tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

#根据指定的维数返回初始化好的指定名称的偏置Variable
def BiasesVariable(shape,name_str,stddev=0.00001):
    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

#2维卷积层（conv2d + bias）封装
def Conv2d(x,W,b,stride=1,padding='SAME'):
    with tf.name_scope('Wx_b'):
        y = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        y = tf.nn.bias_add(y,b)
    return y

#非线性激活层的封装
def Activation(x,activation=tf.nn.relu,name='relu'):
    with tf.name_scope(name):
        y = activation(x)
    return y

#2维池化层pool的封装
def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding='VALID')

#全连接层activate（wx+b）的封装
def FullyConnnected(x,W,b,activate=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x,W)
        y = tf.add(y,b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y

#调用上面写的函数构造计算图
with tf.Graph().as_default():
    #计算图输入
    with tf.name_scope('Inputs'):
        X_origin = tf.placeholder(tf.float32,[None,n_input],name='X_origin')
        Y_true = tf.placeholder(tf.float32, [None, n_class], name='Y_true')
        #把图像数据从N*784的张量转换为N*28*28*1的张量
        X_image = tf.reshape(X_origin,[-1,28,28,1])
    #计算图向前推断过程
    with tf.name_scope('Inference'):
        #第一个卷积层(conv2d + biase)
        with tf.name_scope('Conv2d'):
            weights = WeightsVariable(shape=[5,5,1,16],name_str='weights')
            biases = BiasesVariable(shape=[16],name_str='biases')
            conv_out = Conv2d(X_image,weights,biases,stride=1,padding='VALID')

        #非线性激活层
        with tf.name_scope('Activate'):
            activate_out = Activation(conv_out,activation=tf.nn.relu,name='relu')

        #第一个池化层（max pool 2d）
        with tf.name_scope('Pool2d'):
            pool_out = Pool2d(activate_out,pool=tf.nn.max_pool,k=2,stride=2)

        #将二维特征图变换为一维特征向量
        with tf.name_scope('FeatsReshape'):
            features = tf.reshape(pool_out,[-1,12*12*16])

        #第一个全连接层（fully connected layer）
        with tf.name_scope('FC_Linear'):
            weights = WeightsVariable(shape=[12 * 12 * 16,n_class],name_str='weights')
            biases = BiasesVariable(shape=[n_class],name_str='biases')
            Ypred_logits = FullyConnnected(features,weights,biases,activate=tf.identity,act_name='identity') # activate=tf.identity恒等映射，相当于作了线性转换

    #定义损失层（loss layer）
    with tf.name_scope('Loss'):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_true,logits = Ypred_logits))

    #定义优化训练层（train layer）
    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        trainer = optimizer.minimize(cross_entropy_loss)

    #定义模型评估层(evaluate layer)
    with tf.name_scope('Evaluate'):
        correct_pred = tf.equal(tf.arg_max(Ypred_logits,1),tf.arg_max(Y_true,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    #添加所有变量的初始化节点
    init = tf.global_variables_initializer()

    print('把计算图写入事件文件，在TensorBoard里面查看')
    summary_writer = tf.summary.FileWriter(logdir='../logs',graph=tf.get_default_graph())
    summary_writer.close()