#coding=utf-8

import tensorflow as tf
import numpy as np


## 生成数据
x_data=np.linspace(-0.5, -0.5, 200)[:, np.newaxis]
noise=np.random.normal(0, 0.02, x_data.shape)
y_data=np.square(x_data)+noise

## 定义placeholder
x=tf.placeholder(tf.float32, shape=[None, 1])
y=tf.placeholder(tf.float32, shape=[None, 1])

## 隐藏层
l1_W=tf.Variable(tf.random_normal([1, 10]))
l1_b=tf.Variable(tf.zeros([1, 10]))
l1_output=tf.matmul(x, l1_W)+l1_b
l1_active=tf.nn.relu(l1_output)

l2_W=tf.Variable(tf.random_normal([10, 1]))
l2_b=tf.Variable(tf.zeros([1, 1]))
l2_output=tf.matmul(l1_active, l2_W)+l2_b
y_hat=tf.nn.relu(l2_output)

## loss
loss=tf.reduce_mean(tf.square(y_hat-y))

## train
train=tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

init=tf.global_variables_initializer()


if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0, 1000):
            sess.run(train, feed_dict={x:x_data, y:y_data})
            if i % 100 ==0:
                print(sess.run(loss, feed_dict={x:x_data, y:y_data}))