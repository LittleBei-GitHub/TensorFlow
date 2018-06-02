# coding=utf-8

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义神经网络相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

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
CONV3_DEEP = 120
CONV3_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 84

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE = 0.01
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

mnist = input_data.read_data_sets("../dataset/mnist/", one_hot=True)


# 定义神经网络的前向传播过程
def inference(input_tensor):
    ## 第一层卷积层
    # 和标准LeNet-5模型不大一样，这里定义卷积层的输入为28*28*1的原始MNIST图片像素。
    # 因为卷积层中使用了全0填充，所以输出为28*28*32的矩阵。
    with tf.name_scope('layer1_conv1'):
        conv1_weights = tf.Variable(
            initial_value=tf.truncated_normal(shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                              stddev=0.1,
                                              dtype=tf.float32),
            dtype=tf.float32,
            name='conv1_weights')
        conv1_biases = tf.Variable(
            initial_value=tf.constant(value=0.0, shape=[CONV1_DEEP]),
            dtype=tf.float32,
            name='conv1_biases')
        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1_out = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    ## 第二层池化层
    # 这里选用最大池化层，池化层过滤器的边长为2，使用全0填充且移动的步长为2。
    # 这一层的输入是上一层的输出，也就是28*28*32的矩阵。输出为14*14*32的矩阵。
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ## 第三层卷积层
    # 这一层的输入为14*14*32的矩阵，输出为14*14*64的矩阵。
    with tf.name_scope('layer3_conv2'):
        conv2_weights = tf.Variable(
            initial_value=tf.truncated_normal(shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                              stddev=0.1,
                                              dtype=tf.float32),
            dtype=tf.float32,
            name='conv2_weights')
        conv2_biases = tf.Variable(
            initial_value=tf.constant(value=0.0, shape=[CONV2_DEEP]),
            dtype=tf.float32,
            name='conv2_biases')
        # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2_out = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    ## 第四层池化层
    # 这一层和第二层的结构是一样的。这一层的输入为14*14*64的矩阵，输出为7*7*64的矩阵。
    with tf.name_scope('layer4_pool2'):
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
    conv3_flat = tf.reshape(conv3_out, [-1, nodes])

    ## 第六层全连接层
    with tf.name_scope('layer6_fc1'):
        fc1_weights = tf.Variable(
            initial_value=tf.truncated_normal(shape=[nodes, FC_SIZE],
                                              stddev=0.1,
                                              dtype=tf.float32),
            dtype=tf.float32,
            name="fc1_weights"
        )
        fc1_biases = tf.Variable(
            initial_value=tf.constant(value=0.1, shape=[FC_SIZE]),
            dtype=tf.float32,
            name="fc1_biases"
        )
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, fc1_weights) + fc1_biases)

    ## 第七层输出层
    with tf.name_scope('layer7_fc2'):
        fc2_weights = tf.Variable(
            initial_value=tf.truncated_normal(shape=[FC_SIZE, NUM_LABELS],
                                              stddev=0.1,
                                              dtype=tf.float32),
            dtype=tf.float32,
            name="fc2_weights"
        )
        fc2_biases = tf.Variable(
            initial_value=tf.constant(shape=[NUM_LABELS], value=0.1),
            dtype=tf.float32,
            name="fc2_biases"
        )
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    # 返回第七层的输出
    return logit


if __name__ == '__main__':
    with tf.name_scope("Inputs"):
        # 定义输入输出placeholder
        # 调整输入数据placeholder的格式，输入为一个四维矩阵
        x_images = tf.placeholder(
            dtype=tf.float32,
            shape=[None, INPUT_NODE],  # 第四维表示图片的深度，对于RBG格式的图片，深度为5
            name='X')
        x = tf.reshape(
            tensor=x_images,
            shape=[-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])  # -1:表示暂时不确定
        y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, OUTPUT_NODE],
            name='Y')
    with tf.name_scope("Inference"):
        # 前向传播过程
        y_hat = inference(x)

    with tf.name_scope("Loss"):
        # 定义损失函数、学习率、滑动平均操作以及训练过程
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope("Train"):
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

    with tf.name_scope("Evaluate"):
        correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        total_batches = int(mnist.train.num_examples / BATCH_SIZE)
        print("Per batch Size: ", BATCH_SIZE)
        print("Train sample Count: ", mnist.train.num_examples)
        print("Total batch Count: ", total_batches)
        # 验证和测试的过程将会有一个独立的程序来完成
        for epoch in range(TRAINING_STEPS):
            for i in range(total_batches):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                _, step = sess.run([train_op, global_step], feed_dict={x_images: batch_x, y: batch_y})
                # 每100轮保存一次模型。
                if step % 100 == 0:
                    train_loss, train_eval = sess.run([loss, accuracy], feed_dict={x_images: batch_x, y: batch_y})
                    print(str(step) + '训练集损失：' + str(train_loss))
                    print(str(step) + '训练集评估：' + str(train_eval))

                    validation_loss, validation_eval = sess.run([loss, accuracy],
                                                                feed_dict={x_images: mnist.validation.images,
                                                                           y: mnist.validation.labels})
                    print(str(step) + '验证集损失：' + str(validation_loss))
                    print(str(step) + '验证集评估：' + str(validation_eval))

                    # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                    # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                    # print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                    # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
