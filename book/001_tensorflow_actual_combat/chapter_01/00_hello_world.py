# coding=utf-8

import tensorflow as tf

if __name__ == '__main__':
    # create a constant variable
    hello = tf.constant('hello tensorflow')

    with tf.Session() as sess:
        print(sess.run(hello))
