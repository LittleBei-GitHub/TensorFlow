# coding=utf-8

import tensorflow as tf
import os
import cifar10

image_size = 224
channel_num = 3
class_num = 10
dropout_keep_prob = 0.7
batch_size = 32

num_examples_per_epoch_for_train = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

dataset_dir = '../../dataset/cifar10'
model_save_path = './model/'
model_name = 'model.ckpt'

def get_train_data(data_dir, batch_size):
    cifar10.maybe_download_and_extract(data_dir)
    images, labels = cifar10.get_distorted_train_batch(dataset_dir, batch_size)
    images = tf.image.resize_images(images=images, size=[image_size, image_size])
    return images, labels


def get_eval_data(data_dir, batch_size):
    cifar10.maybe_download_and_extract(data_dir)
    images, labels = cifar10.get_undistorted_eval_batch(dataset_dir, batch_size)
    images = tf.image.resize_images(images=images, size=[image_size, image_size])
    return images, labels

def weights_variable(shape, stddev=0.1, name='weights'):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, name=name)


def biases_variable(shape, name='biases'):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d(input, filter_height, filter_width, num_channels, num_filters, name):
    with tf.name_scope(name):
        weigths = weights_variable(shape=[filter_height, filter_width, num_channels, num_filters],
                                   stddev=0.1)
        biases = biases_variable(shape=[num_filters])
        conv = tf.nn.conv2d(input=input,
                            filter=weigths,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name=name)
        conv_out = tf.nn.relu(features=tf.nn.bias_add(value=conv, bias=biases), name='relu')
        return conv_out


def pool_max(input, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool')


def fc(input, num_in, num_out, name, active=True):
    with tf.name_scope(name):
        weights = weights_variable(shape=[num_in, num_out])
        biases = biases_variable(shape=[num_out])
        fc = tf.matmul(input, weights) + biases
        if active:
            fc_out = tf.nn.relu(features=fc, name='relu')
            return fc_out
        else:
            return fc


def dropout(input, keep_prob, name):
    with tf.name_scope(name):
        return tf.nn.dropout(x=input, keep_prob=keep_prob, name='dropout')


def inference(inputs, train=True):
    with tf.name_scope('conv1'):
        conv1_1 = conv2d(input=inputs, filter_height=3, filter_width=3, num_channels=3, num_filters=64,
                         name='conv1_1')
        conv1_2 = conv2d(input=conv1_1, filter_height=3, filter_width=3, num_channels=64, num_filters=64,
                         name='conv1_2')
        pool1 = pool_max(input=conv1_2, name='pool1')

    with tf.name_scope('conv2'):
        conv2_1 = conv2d(input=pool1, filter_height=3, filter_width=3, num_channels=64, num_filters=128,
                         name='conv2_1')
        conv2_2 = conv2d(input=conv2_1, filter_height=3, filter_width=3, num_channels=128, num_filters=128,
                         name='conv2_2')
        pool2 = pool_max(input=conv2_2, name='pool2')

    with tf.name_scope('conv3'):
        conv3_1 = conv2d(input=pool2, filter_height=3, filter_width=3, num_channels=128, num_filters=256,
                         name='conv3_1')
        conv3_2 = conv2d(input=conv3_1, filter_height=3, filter_width=3, num_channels=256, num_filters=256,
                         name='conv3_2')
        conv3_3 = conv2d(input=conv3_2, filter_height=3, filter_width=3, num_channels=256, num_filters=256,
                         name='conv3_3')
        conv3_4 = conv2d(input=conv3_3, filter_height=3, filter_width=3, num_channels=256, num_filters=256,
                         name='conv3_4')
        pool3 = pool_max(conv3_4, name='pool3')

    with tf.name_scope('conv4'):
        conv4_1 = conv2d(input=pool3, filter_height=3, filter_width=3, num_channels=256, num_filters=512,
                         name='conv4_1')
        conv4_2 = conv2d(input=conv4_1, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv4_2')
        conv4_3 = conv2d(input=conv4_2, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv4_3')
        conv4_4 = conv2d(input=conv4_3, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv4_4')
        pool4 = pool_max(conv4_4, name='pool4')

    with tf.name_scope('conv5'):
        conv5_1 = conv2d(input=pool4, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_1')
        conv5_2 = conv2d(input=conv5_1, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_2')
        conv5_3 = conv2d(input=conv5_2, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_3')
        conv5_4 = conv2d(input=conv5_3, filter_height=3, filter_width=3, num_channels=512, num_filters=512,
                         name='conv5_4')
        pool5 = pool_max(conv5_4, name='pool5')

    with tf.name_scope('fc6') as scope:
        pool5_shape = pool5.get_shape().as_list()
        nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        pool5_flat = tf.reshape(tensor=pool5, shape=[-1, nodes], name='flat')
        fc6 = fc(input=pool5_flat, num_in=nodes, num_out=4096, name=scope)
        if train:
            fc6_dropout = dropout(input=fc6, keep_prob=dropout_keep_prob, name=scope)
        else:
            fc6_dropout = dropout(input=fc6, keep_prob=1.0, name=scope)

    with tf.name_scope('fc7') as scope:
        fc7 = fc(input=fc6_dropout, num_in=4096, num_out=4096, name=scope)
        if train:
            fc7_dropout = dropout(input=fc7, keep_prob=dropout_keep_prob, name=scope)
        else:
            fc7_dropout = dropout(input=fc7, keep_prob=1.0, name=scope)

    with tf.name_scope('fc8') as scope:
        fc8 = fc(input=fc7_dropout, num_in=4096, num_out=class_num, name=scope)

    return fc8


if __name__ == '__main__':
    with tf.name_scope('Train_Batch'):
        train_x, train_y = get_train_data(data_dir=dataset_dir, batch_size=batch_size)

    with tf.name_scope('Valid_Batch'):
        valid_x, valid_y = get_eval_data(data_dir=dataset_dir, batch_size=batch_size)

    with tf.name_scope('Inputs'):
        x = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, image_size, image_size, channel_num],  # 第四维表示图片的深度，对于RBG格式的图片，深度为5
            name='X')
        y = tf.placeholder(
            dtype=tf.int32,  # 不是one-hot时要为整型
            shape=[batch_size],
            name='Y')

    with tf.name_scope('Inference'):
        y_hat = inference(inputs=x)

    with tf.name_scope('Loss'):
        # 在没有one-hot时使用此种损失计算方法
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('Train'):
        global_step = tf.Variable(initial_value=0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss=loss, global_step=global_step)

    with tf.name_scope('Evaluate'):
        top_k = tf.nn.in_top_k(predictions=y_hat, targets=y, k=1, name='top_k')
        # correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(top_k, tf.float32))

    init = tf.global_variables_initializer()
    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        print('===>>>>>>>==开始训练集上训练模型==<<<<<<<=====')
        total_batches = int(num_examples_per_epoch_for_train / batch_size)
        print('Per batch Size:,', batch_size)
        print('Train sample Count Per Epoch:', num_examples_per_epoch_for_train)
        print('Total batch Count Per Epoch:', total_batches)

        # 启动数据读取队列
        tf.train.start_queue_runners()
        for epoch in range(10):
            for i in range(total_batches):
                batch_x, batch_y = sess.run(fetches=[train_x, train_y])
                # print(batch_x.shape())
                _, step = sess.run(fetches=[train, global_step],
                                   feed_dict={x: batch_x, y: batch_y})
                if step % 10 == 0:
                    train_loss, train_eval = sess.run(fetches=[loss, accuracy],
                                                      feed_dict={x: batch_x, y: batch_y})
                    print(str(step) + '训练集损失：' + str(train_loss))
                    print(str(step) + '训练集评估：' + str(train_eval))

                    batch_x, batch_y = sess.run(fetches=[train_x, train_y])
                    validation_loss, validation_eval = sess.run(fetches=[loss, accuracy],
                                                                feed_dict={x: batch_x, y: batch_y})
                    print(str(step) + '验证集损失：' + str(validation_loss))
                    print(str(step) + '验证集评估：' + str(validation_eval))

                    # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                    # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                    # print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                    # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)