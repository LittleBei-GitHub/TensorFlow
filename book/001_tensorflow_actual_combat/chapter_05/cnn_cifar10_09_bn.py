# coding=utf-8

import tensorflow as tf

# 定义batch normalization
def batch_norm(x, iteration, convolutional=False, is_train=True):
    x_shape = x.get_shape().as_list()
    gamma = tf.Variable(initial_value=tf.constant(value=1.0, shape=x_shape[-1]),
                        dtype=tf.float32,
                        name='gamma')
    beta = tf.Variable(initial_value=tf.constant(value=0, shape=x_shape[-1]),
                       dtype=tf.float32,
                       name='beta')
    epsilon = 1e-5
    # 计算均值和方差
    if convolutional:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
    else:
        mean, variance = tf.nn.moments(x, [0], name='moments')
    # 采用滑动平均更新均值与方差
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.999,
                                                       num_updates=iteration)
    def mean_var_with_update():
        ema_apply_op = exp_moving_avg.apply([mean, variance])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(variance)
    # 条件函数
    # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
    m, v = tf.cond(pred=tf.equal(is_train, True),
                   true_fn=mean_var_with_update,
                   false_fn=lambda: (exp_moving_avg.average(mean),
                                     exp_moving_avg.average(variance)))
    #批量正则化
    x_bn = tf.nn.batch_normalization(x=x, mean=m,
                                    variance=v,
                                    offset=beta,
                                    scale=gamma,
                                    variance_epsilon=epsilon)
    return x_bn