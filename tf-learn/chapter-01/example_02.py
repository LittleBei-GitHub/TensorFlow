# coding=utf-8

import sys
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    # 创建一个常亮op，产生一个1*2矩阵，这个op被称作一个节点
    # 加到默认的图中

    # 构造器的返回值代表该常量op的返回值
    matrix_1=tf.constant([[3., 3.]])
    # 创建另外一个常量op，产生一个2*1矩阵
    matrix_2=tf.constant([[2.], [2.]])

    # 创建一个矩阵乘法matmul op， 把‘matrix_1’和‘matrix_2’作为输入
    # 返回值‘product’代表矩阵乘法的结果
    product=tf.matmul(matrix_1, matrix_2)

    # 启动默认的图
    sess=tf.Session()

    # 调用sess的run()方法来执行矩阵乘法op，传入‘product’作为该方法的参数
    # 执行了图中三个op
    result=sess.run(product)
    print(result)

    # 任务完成关闭会话
    sess.close()
