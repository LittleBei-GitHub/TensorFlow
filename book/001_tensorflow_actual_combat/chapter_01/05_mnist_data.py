#coding=utf-8

import tensorflow as tf
## 引入mnist数据集的包
from tensorflow.examples.tutorials.mnist import input_data


if __name__=='__main__':
    print('mnist')

    ## 导入mnist数据集
    mnist=input_data.read_data_sets('mnist_data/', one_hot=True)
    ## 读取训练数据集中的图像
    x_train=mnist.train.images
    ## 读取训练数据集中的标签
    y_train=mnist.train.labels

    ## 读取测试数据集中的图像
    x_test=mnist.test.images
    ## 读取测试数据集中的标签
    y_test=mnist.test.labels

    ## 打印训练数据集的信息
    print('mnist train data images:', x_train.shape)
    print('mnist train data labels:', y_train.shape)

    ## 打印测试数据集的信息
    print('mnist test data images:', x_test.shape)
    print('mnista test data lables', y_test.shape)