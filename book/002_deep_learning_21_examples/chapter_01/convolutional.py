# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets("/home/littlebei/Data/mnist", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# x为训练图像的占位符、y_为训练图像标签的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Dropout中keep_prob的占位符，训练时为0.5，测试时为1
keep_prob = tf.placeholder(tf.float32)

# 将单张图片从784维向量重新还原为28x28的矩阵图片
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# 全连接层，输出为1024维的向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 把1024维的向量转换成10维，对应10个类别
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 直接用tf.nn.softmax_cross_entropy_with_logits直接计算
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# 定义train_step
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义测试的准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    # 创建Session和变量初始化
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 训练20000步
        for step in range(20000):
            batch = mnist.train.next_batch(64)
            sess.run(train, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            # 每100步报告一次在验证集上的准确度
            if step % 100 == 0:
                train_loss, train_accuracy = sess.run(fetches=[cross_entropy, accuracy],
                                                      feed_dict={x: batch[0],
                                                                 y_: batch[1],
                                                                 keep_prob: 1.0})
                validation_loss, validation_accuracy = sess.run(fetches=[cross_entropy, accuracy],
                                                                feed_dict={x: mnist.validation.images,
                                                                           y_: mnist.validation.labels,
                                                                           keep_prob: 1.0})

                print(str(step) + '训练集损失：' + str(train_loss))
                print(str(step) + '训练集评估：' + str(train_accuracy))

                print(str(step) + '验证集损失：' + str(validation_loss))
                print(str(step) + '验证集评估：' + str(validation_accuracy))
        # 训练结束后报告在测试集上的准确度
        print("测试集评估 %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
