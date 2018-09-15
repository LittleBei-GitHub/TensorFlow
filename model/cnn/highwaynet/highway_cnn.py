# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

image_size = 28
channel_num = 1
class_num = 10

learning_rate = 0.01
carry_bias_init = -1.0
keep_prob1 = 0.8
keep_prob2 = 0.7
keep_prob3 = 0.6
keep_prob4 = 0.5

batch_size = 64
train_steps = 100

model_save_path = './model/'
model_name = 'model.ckpt'

mnist = input_data.read_data_sets("../dataset/mnist/", one_hot=True)


def weights_variable(shape, stddev=0.1, name='weights'):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, name=name)


def biases_variable(shape, carry_biases=0.1, name='biases'):
    initial = tf.constant(value=carry_biases, dtype=tf.float32, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d(input, filter_height, filter_width, num_channels, num_filters, strides, padding, activation, name):
    with tf.name_scope(name):
        weigths = weights_variable(shape=[filter_height, filter_width, num_channels, num_filters],
                                   stddev=0.1)
        biases = biases_variable(shape=[num_filters])
        conv = tf.nn.conv2d(input=input,
                            filter=weigths,
                            strides=strides,
                            padding=padding,
                            name=name)
        conv_out = activation(tf.nn.bias_add(value=conv, bias=biases))
        # conv_out = tf.nn.relu(features=tf.nn.bias_add(value=conv, bias=biases), name='relu')
        return conv_out


def conv2d_highway(input, filter_height, filter_width, num_channels, num_filters, strides, padding, activation, name,
                   carry_biases=-0.1):
    with tf.name_scope(name):
        h_weigths = weights_variable(shape=[filter_height, filter_width, num_channels, num_filters],
                                     stddev=0.1)
        h_biases = biases_variable(shape=[num_filters], carry_biases=carry_biases)
        t_weigths = weights_variable(shape=[filter_height, filter_width, num_channels, num_filters],
                                     stddev=0.1)
        t_biases = biases_variable(shape=[num_filters])
        h_conv = tf.nn.conv2d(input=input,
                              filter=h_weigths,
                              strides=strides,
                              padding=padding,
                              name='h_conv')
        t_conv = tf.nn.conv2d(input=input,
                              filter=t_weigths,
                              strides=strides,
                              padding=padding,
                              name='t_conv')
        h = activation(tf.nn.bias_add(value=h_conv, bias=h_biases))
        t = tf.nn.sigmoid(x=tf.nn.bias_add(value=t_conv, bias=t_biases), name='transform_gate')
        c = tf.subtract(x=1.0, y=t, name='carry_gate')
        conv_out = tf.add(x=tf.multiply(h, t), y=tf.multiply(input, c), name='output')
        return conv_out


def fc(input, num_in, num_out, activation, name):
    with tf.name_scope(name):
        weights = weights_variable(shape=[num_in, num_out])
        biases = biases_variable(shape=[num_out])
        fc = tf.matmul(input, weights) + biases
        fc_out = activation(fc)
        return fc_out


def dropout(input, keep_prob, name):
    with tf.name_scope(name):
        return tf.nn.dropout(x=input, keep_prob=keep_prob, name='dropout')


def inference(inputs, train=True):
    with tf.name_scope('layer1'):
        if train:
            dropout1 = tf.nn.dropout(x=inputs, keep_prob=keep_prob1, name='dropout1')
        else:
            dropout1 = tf.identity(input=inputs)
        conv1 = conv2d(input=dropout1,
                       filter_height=5,
                       filter_width=5,
                       num_channels=1,
                       num_filters=32,
                       strides=[1, 1, 1, 1],
                       padding='SAME',
                       activation=tf.nn.relu,
                       name='conv1')
        highway1_1 = conv2d_highway(input=conv1,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway1_1',
                                    carry_biases=carry_bias_init)
        highway1_2 = conv2d_highway(input=highway1_1,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway1_2',
                                    carry_biases=carry_bias_init)
        pool1 = tf.nn.max_pool(highway1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.name_scope('layer2'):
        if train:
            dropout2 = tf.nn.dropout(x=pool1, keep_prob=keep_prob2, name='dropout2')
        else:
            dropout2 = tf.identity(input=pool1)
        highway2_1 = conv2d_highway(input=dropout2,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway2_1',
                                    carry_biases=carry_bias_init)
        highway2_2 = conv2d_highway(input=highway2_1,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway2_2',
                                    carry_biases=carry_bias_init)
        highway2_3 = conv2d_highway(input=highway2_2,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway2_3',
                                    carry_biases=carry_bias_init)
        pool2 = tf.nn.max_pool(highway2_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.name_scope('layer3'):
        if train:
            dropout3 = tf.nn.dropout(x=pool2, keep_prob=keep_prob3, name='dropout3')
        else:
            dropout3 = tf.identity(input=pool2)
        highway3_1 = conv2d_highway(input=dropout3,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway3_1',
                                    carry_biases=carry_bias_init)
        highway3_2 = conv2d_highway(input=highway3_1,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway3_2',
                                    carry_biases=carry_bias_init)
        highway3_3 = conv2d_highway(input=highway3_2,
                                    filter_height=3,
                                    filter_width=3,
                                    num_channels=32,
                                    num_filters=32,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    activation=tf.nn.relu,
                                    name='highway3_3',
                                    carry_biases=carry_bias_init)
        pool3 = tf.nn.max_pool(highway3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        with tf.name_scope('layer4'):
            if train:
                dropout4 = tf.nn.dropout(x=pool3, keep_prob=keep_prob4, name='dropout4')
            else:
                dropout4 = tf.identity(input=pool3)
            dropout4_shape = dropout4.get_shape().as_list()
            nodes = dropout4_shape[1] * dropout4_shape[2] * dropout4_shape[3]
            dropout4_flat = tf.reshape(tensor=dropout4, shape=[-1, nodes], name='flat')
            fc1 = fc(input=dropout4_flat,
                     num_in=nodes,
                     num_out=class_num,
                     activation=tf.identity,
                     name='fc1')
        return fc1


if __name__ == '__main__':
    with tf.name_scope("Inputs"):
        # 定义输入输出placeholder
        # 调整输入数据placeholder的格式，输入为一个四维矩阵
        x_images = tf.placeholder(
            dtype=tf.float32,
            shape=[None, image_size * image_size],  # 第四维表示图片的深度，对于RBG格式的图片，深度为5
            name='X')
        x = tf.reshape(
            tensor=x_images,
            shape=[-1, image_size, image_size, channel_num])  # -1:表示暂时不确定
        y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, class_num],
            name='Y')

    with tf.name_scope('Inference'):
        y_hat = inference(x)

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar(name='loss', tensor=loss)

    with tf.name_scope('Trian'):
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss=loss, global_step=global_step)

    with tf.name_scope("Evaluate"):
        correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar(name='accuracy', tensor=accuracy)

    merged_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir='./logs/', graph=tf.get_default_graph())
    summary_writer.flush()

    init = tf.global_variables_initializer()
    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        total_batches = int(mnist.train.num_examples / batch_size)
        print("Per batch Size: ", batch_size)
        print("Train sample Count: ", mnist.train.num_examples)
        print("Total batch Count: ", total_batches)
        # 验证和测试的过程将会有一个独立的程序来完成
        for epoch in range(train_steps):
            for i in range(total_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, step = sess.run([train, global_step], feed_dict={x_images: batch_x, y: batch_y})
                # 每100轮保存一次模型。
                if step % 100 == 0:
                    train_loss, train_eval, summary = sess.run([loss, accuracy, merged_summaries],
                                                               feed_dict={x_images: batch_x, y: batch_y})
                    summary_writer.add_summary(summary=summary, global_step=step)
                    summary_writer.flush()
                    print(str(step) + '训练集损失：' + str(train_loss))
                    print(str(step) + '训练集评估：' + str(train_eval))

                    validation_loss, validation_eval = sess.run([loss, accuracy],
                                                                feed_dict={x_images: mnist.validation.images,
                                                                           y: mnist.validation.labels})
                    print(str(step) + '验证集损失：' + str(validation_loss))
                    print(str(step) + '验证集评估：' + str(validation_eval))

                    # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                    # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                    # print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                    # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=step)
