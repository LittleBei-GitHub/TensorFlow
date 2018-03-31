#coding=utf-8

import tensorflow as tf


"""1、inference：构建前项预测过程
   2、loss：计算损失
   3、train：训练
   4、eval：评估
"""

x_train=[]
y_train=[]

if __name__=='__main__':
    print('linear')


    with tf.Graph().as_default():
        ## 输入数据（非持久化张量）
        with tf.name_scope('input'):
            X=tf.placeholder(tf.float32, name='X')
            Y=tf.placeholder(tf.float32, name='Y')

        ## 模型参数（持久化张量）
        with tf.name_scope('inference'):
            w=tf.Variable(tf.zeros([1]), name='weight')
            b=tf.Variable(tf.zeros([1]), name='bias')

            ## inference
            y_hat=tf.multiply(X, w)+b

        ## loss
        with tf.name_scope('loss'):
            loss=tf.reduce_sum(tf.pow(y_hat-Y, 2))

        ##train
        with tf.name_scope('train'):
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train=optimizer.minimize(loss)
        ##eval
        # 在线性模型中eval与train相同，省略

        ## 初始化结点
        init=tf.global_variables_initializer()

        ## 保存计算图
        writer=tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
        writer.flush()
        writer.close()

        ## 启动session会话
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(100):
                loss, _=sess.run(fetches=[loss, train], feed_dict={X:x_train, Y:y_train})
                if epoch%10==0:
                    print('loss:',loss)