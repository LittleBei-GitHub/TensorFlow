# coding=utf-8

import cifar10  # 引入tensorflow官方cifar10模块
import tensorflow as tf

# tf.app.flags.FLAGS是TensorFlow内部的一个全局变量存储器
# 同时可以用于命令行参数的处理
FLAGS = tf.app.flags.FLAGS
# 在cifar10模块中预先定义了tf.app.flags.FLAGS.data_dir为CIFAR-10的数据路径
# 我们把这个路径改为cifar10_data
FLAGS.data_dir = '/home/littlebei/Data/cifar10/'

# 如果不存在数据文件，就会执行下载
cifar10.maybe_download_and_extract()
