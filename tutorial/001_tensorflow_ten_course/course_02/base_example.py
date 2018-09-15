#coding=utf-8

import tensorflow as tf
import numpy as np

## 使用numpy生成100个随机点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

w=tf.Variable(0.0) #必须为浮点数
b=tf.Variable(0.0)
y_hat=w*x_data+b

## 损失
loss=tf.reduce_mean(tf.square(y_hat-y_data))

## 优化器
optimizer=tf.train.AdamOptimizer(learning_rate=0.1)

## 训练
train=optimizer.minimize(loss)

## 初始化
init=tf.global_variables_initializer()



if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        for step in range(0, 101):
            sess.run(train)
            if step%20==0:
                print(sess.run(fetches=[w, b]))
