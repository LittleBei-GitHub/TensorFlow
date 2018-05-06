#coding=utf-8

import tensorflow as tf

batch_size=100
n_batch=1000

## 初始化权重
def weight_variable(shape):
    init=tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)

## 初始化偏置
def bais_variable(shape):
    init=tf.constant(value=0.1, shape=shape)
    return tf.Variable(init)

## 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

## 池化层
def max_pool(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## 占位符
x=tf.placeholder(dtype=tf.float32, shape=[None, 784])
y=tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob=tf.placeholder(dtype=32)

## 改变输入图像的格式
x_image=tf.reshape(x, [-1, 28, 28, 1])

## 初始化第一个卷积层
conv1_W=weight_variable(shape=[5, 5, 1, 32])
conv1_b=bais_variable(shape=[32])

conv1_h=tf.nn.relu(conv2d(x=x_image, W=conv1_W)+conv1_b)
pool1_h=max_pool(x=conv1_h)

## 初始化第二个卷积层
conv2_W=weight_variable(shape=[5, 5, 32, 64])
conv2_b=bais_variable(shape=[64])

conv2_h=tf.nn.relu(conv2d(x=pool1_h, W=conv2_W)+conv2_b)
pool2_h=max_pool(x=conv2_h)

## 将层化后的张量展开
pool2_h_flat=tf.reshape(tensor=pool2_h, shape=[-1, 7*7*64])


## 全连接层一
fc1_W=weight_variable(shape=[7*7*64, 1024])
fc1_b=bais_variable([1024])

fc1_h=tf.nn.relu(tf.matmul(a=pool2_h_flat, b=fc1_W)+fc1_b)
fc1_dropout=tf.nn.dropout(x=fc1_h, keep_prob=keep_prob)

## 全连接层二
fc2_W=weight_variable(shape=[1024, 10])
fc2_b=bais_variable(shape=[10])
y_hat=tf.nn.softmax(tf.matmul(a=fc1_dropout, b=fc2_W)+fc2_b)

## 损失
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

## 训练
train=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss)

## 评估
eval=tf.equal(x=tf.argmax(input=y, axis=1), y=tf.argmax(input=y_hat, axis=1))

accuracy=tf.reduce_mean(tf.cast(x=eval, dtype=tf.float32))

init=tf.global_variables_initializer()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(0, 21):
            for batch in range(n_batch):
                sess.run(train, feed_dict={x:None, y:None, keep_prob:0.5})
            sess.run(accuracy, feed_dict={x:None, y:None, keep_prob:1.0})