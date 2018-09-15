# coding=utf-8

import tensorflow as tf

mnist = None
batch_size = 100
n_batch = None

## 占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

## 创建一个单隐藏层网络
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

## loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))

## train
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

## eval
eval = tf.equal(x=tf.argmax(input=y_hat, axis=1), y=tf.argmax(input=y, axis=1))
accuracy = tf.reduce_mean(tf.cast(x=eval, dtype=tf.float32))

saver = tf.train.Saver()

init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(11):
            for batch in range(n_batch):
                train_x = None
                train_y = None
                sess.run(train, feed_dict={x: train_x, y: train_y})
            test_x = None
            test_y = None
            acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        saver.save(sess=sess, save_path='../checkpoint')
