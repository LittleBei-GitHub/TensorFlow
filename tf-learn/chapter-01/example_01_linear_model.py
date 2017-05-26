# coding=utf-8

import sys
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    # 使用numpy生成100个点
    x_data=np.float32(np.random.rand(2, 100))
    y_data=np.dot([0.100, 0.200], x_data)+0.300

    print(x_data)

    # 构造线性模型
    b=tf.Variable(tf.zeros([1]))
    W=tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y=tf.matmul(W, x_data)+b

    # 最小化方差
    loss=tf.reduce_mean(tf.square(y-y_data))
    optimizer=tf.train.GradientDescentOptimizer(0.5)
    train=optimizer.minimize(loss)

    # 初始化变量
    init=tf.initialize_all_variables()

    # 启动图
    session=tf.Session()
    session.run(init)
    # print(session.run(W))

    # 拟合平面
    for step in xrange(0, 100):
        session.run(train)
        if step%20==0:
            print step, session.run(W), session.run(b)

    session.close()