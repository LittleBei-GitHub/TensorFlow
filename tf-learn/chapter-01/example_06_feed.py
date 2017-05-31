# coding=utf-8

import sys
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

if __name__=='__main__':
    input1=tf.placeholder(np.float32)
    input2=tf.placeholder(np.float32)
    output=tf.mul(input1, input2)

    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
        # print(sess.run(input1)) # feed 只在调用它的方法内起作用
