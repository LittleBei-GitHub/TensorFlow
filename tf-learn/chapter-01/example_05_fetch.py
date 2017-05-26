# coding=utf-8

import sys
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    input1=tf.constant(3.0)
    input2=tf.constant(2.0)
    input3=tf.constant(5.0)
    mul=tf.mul(input1, input3)

    sess=tf.Session()
    result=sess.run([input3, mul])
    print(result)
    sess.close()