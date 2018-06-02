# coding=utf-8

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
IMAGE_SIZE=28
IMAGE_SIZE=28
NUM_CHANNELS=3
KEEP_PROB=0.7
NUM_CLASSES=10
LEARNING_RATE=0.01

BATCH_SIZE=64
TRAINING_STEPS=100

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

mnist = input_data.read_data_sets("../dataset/mnist/", one_hot=True)

def conv(x, filter_height, filter_width, num_channels, num_filters, stride_y, stride_x, name,
         padding='SAME'):
    """卷积操作"""
    with tf.name_scope(name) as scope:
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=[filter_height,
                                                                       filter_width,
                                                                       num_channels,
                                                                       num_filters],
                                                                stddev=0.1),
                              dtype=tf.float32,
                              name=scope+'_weights')
        biases = tf.Variable(initial_value=tf.constant(value=0.0,
                                                       shape=[num_filters]),
                             name=scope+'_biases',
                             dtype=tf.float32)
        conv = tf.nn.conv2d(input=x,
                            filter=weights,
                            strides=[1, stride_x, stride_y, 1],
                            padding=padding,
                            name=scope)
        conv_out = tf.nn.relu(features=tf.nn.bias_add(value=conv, bias=biases),
                              name=scope+'_out')
        return conv_out

def fc(x, num_in, num_out, name, active=True):
    """全连接操作"""
    with tf.name_scope(name) as scope:
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=[num_in,
                                                                       num_out],
                                                                stddev=0.1),
                              dtype=tf.float32,
                              name=scope + '_weights')
        biases = tf.Variable(initial_value=tf.constant(value=0.0,
                                                       shape=[num_out]),
                             name=scope + '_biases',
                             dtype=tf.float32)
        full = tf.nn.xw_plus_b(x=x, weights=weights, biases=biases, name=scope)
        if active:
            full_out = tf.nn.relu(features=full, name=scope+'_out')
            return full_out
        else:
            return full

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """池化操作"""
    with tf.name_scope(name) as scope:
        return tf.nn.max_pool(value=x,
                              ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding,
                              name=scope)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """局部响应归一化操作"""
    with tf.name_scope(name) as scope:
        return tf.nn.local_response_normalization(input=x,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=scope)
def dropout(x, keep_prob, name):
    """dropout操作"""
    with tf.name_scope(name) as scope:
        return tf.nn.dropout(x=x,
                             keep_prob=keep_prob,
                             name=scope)


def inference(images):
    """前向推断"""
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    with tf.name_scope('layer1'):
        conv1 = conv(images, 11, 11, 3, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-04, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    with tf.name_scope('layer2'):
        conv2 = conv(pool1, 5, 5, 96, 256, 1, 1, name='conv2')
        norm2 = lrn(conv2, 2, 1e-04, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    with tf.name_scope('layer3'):
        conv3 = conv(pool2, 3, 3, 256, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    with tf.name_scope('layer4'):
        conv4 = conv(conv3, 3, 3, 384, 384, 1, 1, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    with tf.name_scope('layer5'):
        conv5 = conv(conv4, 3, 3, 384, 256, 1, 1, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    with tf.name_scope('layer6'):
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(x=fc6, keep_prob=KEEP_PROB, name='fc6_dropout')

    # 7th Layer: FC (w ReLu) -> Dropout
    with tf.name_scope('layer7'):
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, KEEP_PROB, name='fc7_dropout')

    # 8th Layer: FC and return unscaled activations
    with tf.name_scope('layer8'):
        fc8 = fc(dropout7, 4096, NUM_CLASSES, name='fc8', relu=False)
    return fc8

if __name__=='__main__':
    ## 变量的输入
    with tf.name_scope('Inputs'):
        x_images = tf.placeholder(
            dtype=tf.float32,
            shape=[None, INPUT_NODE],  # 第四维表示图片的深度，对于RBG格式的图片，深度为5
            name='X')
        x = tf.reshape(
            tensor=x_images,
            shape=[-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])  # -1:表示暂时不确定
        y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, NUM_CLASSES],
            name='Y')

    ## 前向推断
    with tf.name_scope('Inference'):
        y_hat = inference(images=x)

    ## 计算损失
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        loss = tf.reduce_mean(cross_entropy)

    ## 训练
    with tf.name_scope('Train'):
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

    ## 评估
    with tf.name_scope("Evaluate"):
        correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        total_batches = int(mnist.train.num_examples / BATCH_SIZE)
        print("Per batch Size: ", BATCH_SIZE)
        print("Train sample Count: ", mnist.train.num_examples)
        print("Total batch Count: ", total_batches)
        # 验证和测试的过程将会有一个独立的程序来完成
        for epoch in range(TRAINING_STEPS):
            for i in range(total_batches):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                _, step = sess.run([train_op, global_step], feed_dict={x_images: batch_x, y: batch_y})
                # 每100轮保存一次模型。
                if step % 100 == 0:
                    train_loss, train_eval = sess.run([loss, accuracy], feed_dict={x_images: batch_x, y: batch_y})
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
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)