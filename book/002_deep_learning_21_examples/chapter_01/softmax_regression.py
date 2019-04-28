# coding=utf-8

import tensorflow as tf
# 导入MNIST教学的模块
from tensorflow.examples.tutorials.mnist import input_data

# 读入MNIST数据
mnist = input_data.read_data_sets("/home/littlebei/Data/mnist", one_hot=True)

# x是一个占位符（placeholder），代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])
# y_也是一个占位符，表示图像的实际标签
y_ = tf.placeholder(tf.float32, [None, 10])

# 权重
W = tf.Variable(tf.zeros([784, 10]))
# 偏置
b = tf.Variable(tf.zeros([10]))

# y=softmax(Wx + b)，y表示模型的预测值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 根据y, y_构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 使用梯度下降法进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# 创建一个Session会话

with tf.Session() as sess:
    # 初始化所有的变量
    sess.run(init)
    print('start training...')

    # 迭代1000轮
    for epoch in range(1000):
        # 每次迭代从数据集中取64张图片
        # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
        # batch_xs, batch_ys对应着两个占位符x和y_
        batch_xs, batch_ys = mnist.train.next_batch(64)
        # 在Session中运行train_step，运行时要传入占位符的值
        loss, _ = sess.run(fetches=[cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})
        print(epoch, ':', loss)

    # 获取最终模型的正确率
    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('accuracy', ':', accuracy_score)
