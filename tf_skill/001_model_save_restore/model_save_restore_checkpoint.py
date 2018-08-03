# coding=utf-8

import tensorflow as tf
import numpy as np
import os

DATA_NUM = 50
EPOCH = 5000
MODEL_SAVE_PATH = './model/model_checkpoint/'
MODEL_NAME = 'model.ckpt'


def generate_fake_data(n):
    """
        生成假数据
    """
    x = np.random.random((n, 1))
    y = x * 4 + 1
    return x, y


def weight_variable(shape, stddev=0.1, name='weight'):
    """
        定义权重变量
    """
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, name=name)


def bias_variable(shape, name='biase'):
    """
        定义偏置变量
    """
    initial = tf.constant(value=0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial_value=initial, name=name)


def inference(input):
    """
        前向推断过程
    """
    w = weight_variable([1])
    b = bias_variable([1])
    y_hat = tf.nn.bias_add(value=tf.multiply(input, w), bias=b, name='y_hat')
    return y_hat


def save_model():
    """
        保存模型
    """
    # 占位符
    input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
    label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
    # 预测数据
    y_hat = inference(input)
    # 损失
    loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(y_hat, label), name='loss'))
    # 创建优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01, name='optimizer')
    # 训练操作
    train = optimizer.minimize(loss=loss, name='train')
    # 初始化全局变量
    init = tf.global_variables_initializer()

    # 创建模型保存器
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        x, y = generate_fake_data(DATA_NUM)

        for i in range(EPOCH + 1):
            train_loss, _ = sess.run(fetches=[loss, train], feed_dict={input: x, label: y})
            # 每100次输出一次损失结果
            if i % 100 == 0:
                print('loss:' + str(train_loss))
                # 每循环100次保存一次模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i, write_meta_graph=False)


def restore_model():
    """
        恢复模型
    """
    # 占位符
    input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
    # 预测数据
    y_hat = inference(input)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 注意：如果在保存模型阶段使用了global_step属性，则在模型保存的结果中会默认出现5个.index和.data-00000-of-00001文件
        # 并且在你的模型名与.index之间会出现'-数字'的形式（.data-00000-of-00001也一样），
        # 其中数字最大的为最新的模型参数，
        # 这个时候在恢复模型时除了模型路径和你命名的模型名字以外还要加上最新的数字，
        # 否则会报找不到模型的错误。

        # 加载模型
        saver.restore(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME+'-5000'))
        y = sess.run(y_hat, feed_dict={input: [[3]]})
        print(y)


if __name__ == '__main__':
    # save_model()
    restore_model()
