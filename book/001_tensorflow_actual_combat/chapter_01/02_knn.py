#coding=utf-8

"""KNN model
   三要素：
   1、距离度量
   2、k值的选择
   3、分类决策规则
"""

"""1、inference()构建模型的前向预测过程
   2、evaluate()在测试集上对模型的预测性能进行评估
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



if __name__=='__main__':
    print('knn model')
    mnist=input_data.read_data_sets('mnist_data/', one_hot=True)

    # 取mnist数据集中的1000个样本用作训练集
    Xtrain, Ytrain=mnist.train.next_batch(1000)
    Xtest, Ytest=mnist.test.next_batch(100)

    # 输出样本数据集的信息
    print('Xtrain.shape:', Xtrain.shape)
    print('Ytrain.shape:', Ytrain.shape)

    print('Xtest.shape:', Xtest.shape)
    print('Ytest.shape:', Ytest.shape)

    # 训练数据集占位符
    xtrain=tf.placeholder(tf.float32, [None, 784])
    ytrain=tf.placeholder(tf.float32, [None, 10])

    # 测试数据集占位符
    xtest=tf.placeholder(tf.float32, [784])

    # 计算新样本与训练数据集之间的距离
    distance=tf.reduce_sum(tf.abs(tf.subtract(xtest, xtrain)), axis=1)
    # 找出距离最小的索引
    y_hat_index=tf.arg_min(distance, 0)
    #pre_class=tf.arg_max(Ytrain[y_hat_index], 1)

    # 初始化变量
    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(0, len(Xtest)):
            pre_index=sess.run(y_hat_index, feed_dict={xtrain:Xtrain, ytrain:Ytrain, xtest:Xtest[i, :]})
            # 根据索引值找到y_hat从而找出预测类别
            pre_class=np.argmax(Ytrain[pre_index])
            # 真实类别
            true_class=np.argmax(Ytest[i])
            print('test sample:', i, 'true_class:', true_class, 'pre_class:', pre_class)