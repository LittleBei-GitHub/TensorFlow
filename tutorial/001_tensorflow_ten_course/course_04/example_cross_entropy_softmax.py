# coding=utf-8

import tensorflow as tf

x_data = None
y_data = None

## placehodler
x = tf.placeholder(type=tf.float32, shape=[None, 784])
y = tf.placeholder(type=tf.float32, shape=[None, 10])

## infer
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

## loss
# loss=tf.reduce_mean(tf.square(y_hat-y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

## train
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

## eval
eval = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(eval, tf.float32))

## init
init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(0, 101):
            sess.run(train, feed_dict={x: x_data, y: y_data})
