# coding=utf-8

import sys
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    # 创建一个变量，初始化标量0
    counter=tf.Variable(0)

    # 创建一个op，其作用是使counter加1
    one=tf.constant(1)
    new_value=tf.add(counter, one)
    update=tf.assign(counter, new_value)

    # 启动图后，变量必须先经过init op初始化
    # 首先必须增加一个‘初始化’的op到图中
    init_op=tf.initialize_all_variables()

    # 创建session会话，启动图
    with tf.Session() as sess:
        sess.run(init_op)
        # 打印‘counter’初始值
        print sess.run(counter)
        for _ in range(3):
            sess.run(update)
            print sess.run(counter)