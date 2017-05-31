# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf


reload(sys)
sys.setdefaultencoding('utf-8')


if __name__=='__main__':
    mnist=input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_train_images=mnist.train.images
    mnist_train_labels=mnist.train.labels
    print(len(mnist_train_images))