#coding=utf-8

import tensorflow as tf


if __name__=='__main__':
    ## base operation
    # constant
    a=tf.constant(1)
    b=tf.constant(2)

    with tf.Session() as sess:
        print(sess.run(a+b))
        print(sess.run(a*b))

    # placeholder
    # need tensorflow inter function
    a=tf.placeholder(tf.int8)
    b=tf.placeholder(tf.int8)

    add=tf.add(a, b)
    mul=tf.multiply(a, b)
    with tf.Session() as sess:
        print(sess.run(add, feed_dict={a:3, b:4}))
        print(sess.run(mul, feed_dict={a:3, b:4}))

    # constant matrix
    m1=tf.constant([[3, 3]], tf.float16)
    m2=tf.constant([[2], [2]], tf.float16)
    # matrix multiply
    matrix_mul=tf.matmul(m1, m2)
    with tf.Session() as sess:
        print(sess.run(matrix_mul))


    ## save compute graph
    writer=tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
    writer.flush()
