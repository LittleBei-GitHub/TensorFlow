#coding=utf-8

import tensorflow as tf

mnist=None     # 数据集
n_inputs=28    # 输入一行，一行数据有28个分量
max_time=28    # 总共有28行
lstm_size=100  # 隐藏单元
n_class=10     # 10个类
batch_size=50  # 每个批次有50个样本
n_batch=None   # 批次的数量

## 占位符
x=tf.placeholder(dtype=tf.float32, shape=[None, 784])
y=tf.placeholder(dtype=tf.float32, shape=[None, 10])

## 初始化权重
weights=tf.Variable(tf.truncated_normal(shape=[lstm_size, n_class], stddev=0.1), name='w')
## 初始化偏置
biases=tf.Variable(tf.constant(0.1, shape=[n_class]), name='b')

## 定义网络
def rnn(X, weights, biases):
    inputs=tf.reshape(tensor=X, shape=[-1, max_time, n_inputs])
    #lstm 基本单元
    lstm_cell=tf.contrib.rnn.core_rnn_cell.BasicaLSTMCell(lstm_size)
    outputs, final_state=tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inputs, dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1], weights)+biases)
    return results

## 输出结果
y_hat=rnn(x, weights, biases)

## loss
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))

## train
train=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss)

## eval
eval=tf.equal(x=tf.argmax(input=y_hat, axis=1), y=tf.argmax(input=y, axis=1))
accuracy=tf.reduce_mean(tf.cast(x=eval, dtype=tf.float32))

init=tf.global_variables_initializer()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(6):
            for batch in range(n_batch):
                sess.run(train, feed_dict={x:None, y:None})
            test_x=None
            test_y=None
            acc=sess.run(accuracy, feed_dict={x:test_x, y:test_y})