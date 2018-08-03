# coding=utf-8

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import os

DATA_NUM = 50
EPOCH = 5000
MODEL_SAVE_PATH = './model/model_pb_1/'
MODEL_NAME = 'model.pb'


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

        # 模型保存
        # convert_variables_to_constants 需要指定 output_node_names，list()，可以多个
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y_hat'])

        # 写入序列化的pb文件
        with tf.gfile.FastGFile(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def restore_model():
    """
        恢复模型
    """
    # 创建session会话
    with tf.Session() as sess:
        # 模型加载
        with gfile.FastGFile(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

        # 通过变量的name属性名字来获取变量
        x = sess.graph.get_tensor_by_name('x:0')
        y_hat = sess.graph.get_tensor_by_name('y_hat:0')

        # 运行操作结果获取结果
        y = sess.run(y_hat, feed_dict={x: [[3]]})
        print(y)


if __name__ == '__main__':
    # save_model()
    restore_model()
