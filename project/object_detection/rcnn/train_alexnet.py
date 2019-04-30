from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
from PIL import Image
import os.path

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def load_image(img_path):
    """
        读取单张图片
    """
    img = Image.open(img_path)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """
        调整图片大小
    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def pil_to_nparray(pil_image):
    """
        将图片转成数组格式
    """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def load_data(datafile, num_clss, save=False, save_path='dataset.pkl'):
    """
        加载数据集
    """
    # 打开存放训练数据文件名的文件
    train_list = open(datafile, 'r')
    # 标签
    labels = []
    # 图片
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        # 获取图片
        fpath = tmp[0]
        print(fpath)
        img = load_image(fpath)
        img = resize_image(img, 224, 224)  # 调整图片大小为224
        np_img = pil_to_nparray(img)
        images.append(np_img)
        # 获取图片索引
        index = int(tmp[1])
        label = np.zeros(num_clss)
        label[index] = 1  # 转成one-hot编码的格式
        labels.append(label)
    # 持久化到文件中
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_from_pkl(dataset_file):
    """
        从pickle文件中加载数据集
    """
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X, Y


def create_alexnet(num_classes):
    """
        Alexnet网络结构
    """
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def train(network, X, Y):
    """
        训练
    """
    # Training
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output')
    if os.path.isfile('model_save.model'):
        model.load('model_save.model')
    model.fit(X, Y, n_epoch=100, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17')  # epoch = 1000
    # Save the model
    model.save('model_save.model')


def predict(network, modelfile, images):
    """
        预测
    """
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)


if __name__ == '__main__':
    X, Y = load_data('train_list.txt', 17)
    # X, Y = load_from_pkl('dataset.pkl')
    net = create_alexnet(17)
    train(net, X, Y)
