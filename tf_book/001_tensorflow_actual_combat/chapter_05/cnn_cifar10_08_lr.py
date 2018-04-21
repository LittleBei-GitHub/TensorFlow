#coding=utf-8
## 实现简单卷积神经网络对MNIST数据集进行分类
## conv2d + activation + pool + fc
## learning rate
import csv
import tensorflow as tf


# 设置算法超参数
training_epochs = 5
num_examples_per_epoch_for_train = 10000
batch_size = 100
learning_rate_init = 0.1
learning_rate_final = 0.001
learning_rate_decay_rate = 0.7
# learning_rate_decay_rate = 0.5
num_batches_per_epoch = int(num_examples_per_epoch_for_train/batch_size)
num_epochs_per_decay = 1 #Epochs after which learning rate decays
learning_rate_decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

display_step = 50
conv1_kernel_num = 32
conv2_kernel_num = 32
fc1_units_num = 250
fc2_units_num = 150
activation_func = tf.nn.relu
activation_name = 'relu'
l2loss_ratio = 0.05

with tf.Graph().as_default():
    #优化器调用次数计算器，全局训练步数
    global_step = tf.Variable(0,name='global_step',trainable=False,dtype=tf.int64)

    #使用exponential_decay产生指衰减的学习率
    learning_rate = tf.train.exponential_decay(learning_rate_init,
                                               global_step,
                                               learning_rate_decay_steps,
                                               learning_rate_decay_rate,
                                               staircase=False)

    #使用polynomial_decay产生多项式衰减的学习率
    # learning_rate = tf.train.polynomial_decay(learning_rate_init,
    #                                             global_step,
    #                                             learning_rate_decay_steps,
    #                                             learning_rate_decay_rate,
    #                                             staircase=False)

    # 使用natural_exp_decay产生自然指数衰减的学习率
    # learning_rate = tf.train.natural_exp_decay(learning_rate_init,
    #                                           global_step,
    #                                           learning_rate_decay_steps,
    #                                           learning_rate_decay_rate,
    #                                           staircase=False)

    # 使用inverse_time_decay产生逆时间衰减的学习率
    # learning_rate = tf.train.inverse_time_decay(learning_rate_init,
    #                                            global_step,
    #                                            learning_rate_decay_steps,
    #                                            learning_rate_decay_rate,
    #                                            staircase=False)

    #定义损失函数
    weights = tf.Variable(tf.random_normal([9000,9000],mean=0.0,stddev=1e9,dtype=tf.float32))
    myloss = tf.nn.l2_loss(weights,name="L2Loss")

    #传入 learning_rate创建优化器对象
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #将global_step传入minimize()，每次调用minimize都会使得global_step自增1.
    training_op = optimizer.minimize(myloss,global_step=global_step)

    #添加所有变量的初始化节点
    init_op = tf.global_variables_initializer()

    #将评估结果保存到文件
    results_list = list()
    results_list.append(['train_step','learning_rate','train_step','train_loss'])

    #启动会话，训练模型
    with tf.Session() as sess:
        sess.run(init_op)
        #训练指定轮数，每一轮的训练样本总数为：num_examples_per_epoch_for_train
        for epoch in range(training_epochs):
            print('********************************************')
            #每一轮都要把所有的batch跑一遍
            for batch_idx in range(num_batches_per_epoch):
                #获取learning_rate的值
                current_learning_rate = sess.run(learning_rate)
                #执行训练节点，获取损失节点和global_step的值
                _,loss_value,training_step = sess.run([training_op,myloss,global_step])

                print("Training Epoch:" + str(epoch) +
                      ",Training Step:" + str(training_step)+
                      ",Learning Rate = " + "{:.6f}".format(current_learning_rate) +
                      ",Learning Loss = " + "{:.6f}".format(loss_value))
                #记录结果
                results_list.append([training_step,current_learning_rate,training_step,loss_value])

    #将评估结果保存到文件
    print("训练结束，将结果保存到文件")
    results_file = open('evaluate_results/learning_rate/exponential_decay/evaluate_results(decay_rate=0.7,staircase=False).csv','w',newline='')
    csv_writer = csv.writer(results_file,dialect='excel')
    for row in results_list:
        csv_writer.writerow(row)