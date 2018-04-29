#coding=utf-8

import tensorflow as tf

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

add=tf.add(x, y)

if __name__=='__main__':
    with tf.Session() as sess:
        rs=sess.run(add, feed_dict={x:1.0, y:2.0})
        print(rs)