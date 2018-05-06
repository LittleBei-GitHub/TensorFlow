#coding=utf-8

import tensorflow as tf

batch_size=100
n_batch=1000

## 参数总结
def variable_summaries(var):
    with tf.name_scope(name='summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar(name='mean', tensor=mean)
        with tf.name_scope(name='sttdev'):
            sttdev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar(name='sttdev', tensor=sttdev)
        tf.summary.scalar(name='max', tensor=tf.reduce_max(var))
        tf.summary.scalar(name='min', tensor=tf.reduce_min(var))
        tf.summary.histogram(name='histogram', values=var)


## 初始化权重
def weight_variable(shape, name):
    init=tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init, name=name)

## 初始化偏置
def bais_variable(shape, name):
    init=tf.constant(value=0.1, shape=shape)
    return tf.Variable(init, name=name)

## 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

## 池化层
def max_pool(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope(name='input'):
    ## 占位符
    x=tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
    y=tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
    keep_prob=tf.placeholder(dtype=32, name='keep_prob')
    with tf.name_scope(name='x_image'):
        ## 改变输入图像的格式
        x_image=tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope(name='Conv1'):
    ## 初始化第一个卷积层
    with tf.name_scope(name='W'):
        conv1_W=weight_variable(shape=[5, 5, 1, 32], name='conv1_W')
    with tf.name_scope(name='b'):
        conv1_b=bais_variable(shape=[32], name='conv1_b')
    with tf.name_scope(name='conv'):
        conv1_h=tf.nn.relu(conv2d(x=x_image, W=conv1_W)+conv1_b)
    with tf.name_scope(name='pool'):
        pool1_h=max_pool(x=conv1_h)

with tf.name_scope(name='Conv2'):
    ## 初始化第二个卷积层
    with tf.name_scope(name='W'):
        conv2_W=weight_variable(shape=[5, 5, 32, 64], name='conv2_W')
    with tf.name_scope(name='b'):
        conv2_b=bais_variable(shape=[64], name='conv2_b')
    with tf.name_scope(name='conv'):
        conv2_h=tf.nn.relu(conv2d(x=pool1_h, W=conv2_W)+conv2_b)
    with tf.name_scope(name='pool'):
        pool2_h=max_pool(x=conv2_h)

with tf.name_scope(name='fc1'):
    with tf.name_scope(name='flat'):
        ## 将层化后的张量展开
        pool2_h_flat=tf.reshape(tensor=pool2_h, shape=[-1, 7*7*64])
    ## 全连接层一
    with tf.name_scope(name='W'):
        fc1_W=weight_variable(shape=[7*7*64, 1024], name='fc1_W')
    with tf.name_scope(name='b'):
        fc1_b=bais_variable([1024], name='fc1_b')
    with tf.name_scope(name='fc'):
        fc1_h=tf.nn.relu(tf.matmul(a=pool2_h_flat, b=fc1_W)+fc1_b)
    with tf.name_scope(name='dropout'):
        fc1_dropout=tf.nn.dropout(x=fc1_h, keep_prob=keep_prob)

with tf.name_scope(name='fc2'):
    ## 全连接层二
    with tf.name_scope(name='W'):
        fc2_W=weight_variable(shape=[1024, 10], name='fc2_W')
    with tf.name_scope(name='b'):
        fc2_b=bais_variable(shape=[10], name='fc2_b')
    with tf.name_scope(name='softmax'):
        y_hat=tf.nn.softmax(tf.matmul(a=fc1_dropout, b=fc2_W)+fc2_b)

with tf.name_scope(name='loss'):
    ## 损失
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    tf.summary.scalar(name='loss', tensor=loss)

with tf.name_scope(name='train'):
    ## 训练
    train=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss)

with tf.name_scope(name='eval'):
    ## 评估
    eval=tf.equal(x=tf.argmax(input=y, axis=1), y=tf.argmax(input=y_hat, axis=1))
    accuracy=tf.reduce_mean(tf.cast(x=eval, dtype=tf.float32))
    tf.summary.scalar(name='accuracy', tensor=accuracy)

merged=tf.summary.merge_all()

init=tf.global_variables_initializer()


if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        train_writer=tf.summary.FileWriter(logdir='../logs/train', graph=sess.graph)
        test_writer=tf.summary.FileWriter(logdir='../logs/test', graph=sess.graph)
        for epoch in range(0, 21):
            sess.run(train, feed_dict={x:None, y:None, keep_prob:0.5})
            ## train
            train_x=None
            train_y=None
            summary=sess.run(merged, feed_dict={x:train_x, y:train_y, keep_prob:1.0})
            train_writer.add_summary(summary)

            ## test
            test_x=None
            test_y=None
            summary = sess.run(merged, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
            test_writer.add_summary(summary)

            for batch in range(n_batch):
                sess.run(train, feed_dict={x:None, y:None, keep_prob:0.5})
            sess.run(accuracy, feed_dict={x:None, y:None, keep_prob:1.0})