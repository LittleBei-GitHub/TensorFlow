#coding=utf-8

import tensorflow as tf


input1=tf.constant(1.0)
input2=tf.constant(2.0)
input3=tf.constant(3.0)

add=tf.add(input1, input2)
mul=tf.multiply(input1, input2)

if __name__=='__main__':
    with tf.Session() as sess:
        rs=sess.run(fetches=[add, mul])
        print(rs)