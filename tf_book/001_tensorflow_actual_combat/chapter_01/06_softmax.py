#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


"""softmax"""
"""mnist"""

IMAGE_SIZE=784
LABEL_SIZE=10
BATCH_SIZE=100

mnist=input_data.read_data_sets('mnist_data/', one_hot=True)

if __name__=='__main__':
    print('softmax')

    with tf.Graph().as_default():
        ## 输入节点
        with tf.name_scope('input'):
            X=tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE], name='X')
            Y=tf.placeholder(tf.float32, shape=[None, LABEL_SIZE], name='Y')

        ## 前向推断
        with tf.name_scope('inference'):
            W=tf.Variable(tf.zeros([IMAGE_SIZE, LABEL_SIZE]), name='Weights')
            b=tf.Variable(tf.zeros([LABEL_SIZE]), name='Biases')
            logits=tf.add(tf.matmul(X, W), b)

            ## softmax
            with tf.name_scope('softmax'):
                Y_hat=tf.nn.softmax(logits=logits)

        ## loss
        with tf.name_scope('loss'):
            loss=tf.reduce_mean(-tf.reduce_sum(tf.multiply(Y, tf.log(Y_hat)), axis=1))

        ## train
        with tf.name_scope('train'):
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train=optimizer.minimize(loss)

        ## eval
        with tf.name_scope('evaluation'):
            correct_prediction=tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1))
            eval=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast()tf中的类型转换函数

        ## initializer
        init=tf.global_variables_initializer()

        ## 保存计算图
        writer=tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
        writer.flush()
        writer.close()


        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(100):
                X_train, Y_train=mnist.train.next_batch(BATCH_SIZE)
                loss_score, _ = sess.run(fetches=[loss, train], feed_dict={X:X_train, Y:Y_train})
                print(epoch, ':', loss_score)

            accuracy_score=sess.run(eval, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
            print('eval:', accuracy_score)
