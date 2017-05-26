# coding=utf-8

import sys
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    # 进入一个交互式的TensorFlow会话
    sess=tf.InteractiveSession()
    x=tf.Variable([1.0, 2.0])
    a=tf.constant([3.0, 3.0])

    #使用初始化器initializer op 的run()方法初始化‘x’
    x.initializer.run()

    sub=tf.sub(x, a)
    print sub.eval()