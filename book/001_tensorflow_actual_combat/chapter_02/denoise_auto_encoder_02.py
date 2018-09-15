# coding=utf-8

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


## version1.2
## 添加训练操作

## Xaiver 均匀初始化
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


## 加性高斯噪声的自动编码器
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        self.training_scale = scale
        self.weights = dict()

        with tf.name_scope('RawInput'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
        with tf.name_scope('NoiseAdder'):
            self.scale = tf.placeholder(tf.float32)
            self.x_noise = self.x + self.scale * tf.random_normal((n_input,))
        with tf.name_scope('Encoder'):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
            self.hidden = self.transfer(tf.add(tf.matmul(self.x_noise, self.weights['w1']), self.weights['b1']))
        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32)
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)
            self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        with tf.name_scope('Loss'):
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))
        with tf.name_scope('Train'):
            self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('begin to run session....')

    def partial_fit(self, X):  # 训练优化
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                          self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):  # 计算损失，不优化
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):  # 压缩维度
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):  # 低维变高维
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, fedd_dict={self.hidden: hidden})

    def reconstruct(self, X):  #
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weight['w1'])

    def getBiaes(self):
        return self.sess.run(self.weight['b1'])

## 数据标准化
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


if __name__ == '__main__':
    training_epochs = 20
    batch_size = 128
    display_step = 1
    mnist = input_data.read_data_sets("../../mnist_data/", one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train._num_examples)

    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input=786, n_hidden=200,
                                                   transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                                   scale=0.01)
    print('把计算图写入时间文件，在TensorBoard里面查看')
    writer_summary = tf.summary.FileWriter(logdir='../logs', graph=autoencoder.sess.graph)
    writer_summary.close()

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
