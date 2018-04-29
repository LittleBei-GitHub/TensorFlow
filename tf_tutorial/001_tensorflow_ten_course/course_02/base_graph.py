#coding=utf-8

import tensorflow as tf

if __name__=='__main__':
    constant1=tf.constant([[3, 3]])
    constant2=tf.constant([[2], [3]])
    mt=tf.matmul(constant1, constant2)

    with tf.Session() as sess:
        rs=sess.run(mt)
        print(rs)