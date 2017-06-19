# coding=utf-8

import tensorflow as tf
import numpy as np


if __name__=='__main__':
    # 初始化一个变量
    weights=tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='weights')
    biases=tf.Variable(tf.zeros([200]), name='biases')

    # 用一个变量初始化另一个变量
    w2=tf.Variable(weights.initialized_value(), name='w2')
    w_twice=tf.Variable(weights.initialized_value()*0.2, name='w_twice')

    init_op=tf.initialize_all_variables()

    # 保存变量
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print sess.run(weights)
        save_path=saver.save(sess, '/data0/shaojie5/littlebei/program/python/pycharm/TensorFlow/data/model.ckpt')
        print 'model saved in file:', save_path

    # 加载变量
    with tf.Session() as sess:
        saver.restore(sess, '/data0/shaojie5/littlebei/program/python/pycharm/TensorFlow/data/model.ckpt')
        print 'model restored'
        print sess.run(weights)