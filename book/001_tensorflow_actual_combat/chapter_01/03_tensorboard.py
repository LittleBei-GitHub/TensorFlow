#coding=utf-8

import tensorflow as tf

"""使用Summary来汇总数据"""
"""使用name_scope来划分图层"""

IMAGE_PIXELS=10


if __name__=='__main__':
    print('tensorboard')

    ## 测试name_scope的效果

    image=tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))

    hidden1_units=20
    with tf.name_scope('hidden1'): # name_scope结点
        weights=tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=0.3), name='weights')
        biases=tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1=tf.nn.relu(tf.matmul(image, weights)+biases)

    hidden2_units = 10
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=0.3), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)


    writer=tf.summary.FileWriter('../logs', tf.get_default_graph())
    writer.flush()
    writer.close()
    # with tf.Session() as sess:
    #     sess