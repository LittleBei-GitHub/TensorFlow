# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

input_layer_size = 784
hidden_layer_size = 50 # use ~71 for fully-connected (plain) layers, 50 for highway layers
output_layer_size = 10
layer_count = 20

image_size = 28
channel_num = 1
class_num = 10

learning_rate = 0.01
carry_bias_init = -2.0

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


def fc(input, num_in, num_out, activation, name):
    with tf.name_scope(name):
        weights = weights_variable(shape=[num_in, num_out])
        biases = biases_variable(shape=[num_out])
        fc = tf.matmul(input, weights) + biases
        fc_out = activation(fc)
        return fc_out

def highway(input, num_in, num_out, activation, name, carry_biases=-1.0):
    with tf.name_scope(name):
        h_weights = weights_variable(shape=[num_in, num_out])
        h_biases = biases_variable(shape=[num_out])

        t_weights = weights_variable(shape=[num_in, num_out])
        t_biases = biases_variable(shape=[num_out], carry_biases=carry_biases)

        h = activation(tf.matmul(a=input, b=h_weights)+h_biases, name='activation')
        t = tf.nn.sigmoid(tf.matmul(a=input, b=t_weights)+t_biases, name='transform_gate')
        c = tf.subtract(1.0, t, name='carry_gate')
        out = tf.add(tf.multiply(h, t), tf.multiply(input, c), name='highway_out')
        return out


def inference(input):
    layer_out = None
    y_hat = None
    for i in range(layer_count):
        with tf.name_scope("layer{0}".format(i)) as scope:
            if i == 0:  # first, input layer
                layer_out = fc(input, input_layer_size, hidden_layer_size, tf.nn.relu, name=scope)
            elif i == layer_count - 1:  # last, output layer
                y_hat = fc(layer_out, hidden_layer_size, output_layer_size, tf.nn.softmax, name=scope)
            else:  # hidden layers
                layer_out = highway(layer_out, hidden_layer_size, hidden_layer_size, tf.nn.relu, name=scope, carry_biases=carry_bias_init)
    return y_hat


if __name__ == '__main__':
    with tf.name_scope("Inputs"):
        # 定义输入输出placeholder
        # 调整输入数据placeholder的格式，输入为一个四维矩阵
        x_images = tf.placeholder(
            dtype=tf.float32,
            shape=[None, image_size * image_size],  # 第四维表示图片的深度，对于RBG格式的图片，深度为5
            name='X')
        # x = tf.reshape(
        #     tensor=x_images,
        #     shape=[-1, image_size, image_size, channel_num])  # -1:表示暂时不确定
        y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, class_num],
            name='Y')

    with tf.name_scope('Inference'):
        y_hat = inference(x_images)

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