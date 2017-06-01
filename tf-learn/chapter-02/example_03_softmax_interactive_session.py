# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    ## 导入数据集
    mnist=input_data.read_data_sets('MNIST_data/', one_hot=True)

    ## 开启交互式session
    sess=tf.InteractiveSession()

    ## 占位符
    x=tf.placeholder(tf.float32, shape=[None, 784])
    y_=tf.placeholder(tf.float32, shape=[None, 10])

    ## 参数初始化
    W=tf.Variable(tf.zeros([784, 10]))
    b=tf.Variable(tf.zeros([10]))

    sess.run(tf.initialize_all_variables())

    ## 定义激活函数
    y=tf.nn.softmax(tf.matmul(x, W)+b)
    ## 定义损失函数
    cross_entropy=-tf.reduce_sum(y_*tf.log(y))
    # loss=tf.reduce_mean(cross_entropy)

    ## 模型训练
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
        batch=mnist.train.next_batch(50)
        train_step.run(feed_dict={x:batch[0], y_:batch[1]})
        print cross_entropy.eval(feed_dict={x:batch[0], y_:batch[1]})
        # print loss.eval(feed_dict={x:batch[0], y_:batch[1]})


    ## 模型评估
    correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))