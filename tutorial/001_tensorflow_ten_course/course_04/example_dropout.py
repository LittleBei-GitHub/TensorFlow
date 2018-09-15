# coding=utf-8

import tensorflow as tf

x_data = None
y_data = None

## placehodler
x = tf.placeholder(type=tf.float32, shape=[None, 784])
y = tf.placeholder(type=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(type=tf.float32)

## infer
W1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]) + 0.1)
l1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
l1_dropout = tf.nn.dropout(l1_output, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([100, 200], stddev=0.1))
b2 = tf.Variable(tf.zeros([200]) + 0.1)
l2_output = tf.nn.relu(tf.matmul(l1_dropout, W2) + b2)
l2_dropout = tf.nn.dropout(l2_output, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b3 = tf.Variable(tf.zeros([100]) + 0.1)
l3_output = tf.nn.relu(tf.matmul(l2_dropout, W3) + b3)
l3_dropout = tf.nn.dropout(l3_output, keep_prob=keep_prob)

W4 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
y_hat = tf.nn.softmax(tf.matmul(l3_dropout, W4) + b4)

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
            sess.run(train, feed_dict={x: x_data, y: y_data, keep_prob: 0.7})
