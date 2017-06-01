# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    mnist=input_data.read_data_sets('MNIST_data/', one_hot=True)

    ## 实现回归模型
    x=tf.placeholder(tf.float32, [None, 784])
    y_=tf.placeholder(tf.float32, [None, 10])

    W=tf.Variable(tf.zeros([784, 10]))   # 回归系数W
    b=tf.Variable(tf.zeros([10]))        # 偏置b

    y=tf.nn.softmax(tf.matmul(x, W)+b)   # 激活函数

    ## 训练模型
    cross_entropy=-tf.reduce_sum(y_*tf.log(y))  # 损失函数
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init=tf.initialize_all_variables()

    sess=tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys=mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

    ## 模型评估
    correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

    sess.close()