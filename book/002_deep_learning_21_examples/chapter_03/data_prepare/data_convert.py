# coding=utf-8
from __future__ import absolute_import
import argparse
import os
import logging
from tfrecord import main


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据存放目录
    parser.add_argument('-t', '--tensorflow_data_dir', default='pic/')
    parser.add_argument('--train_shards', default=2, type=int)
    parser.add_argument('--validation_shards', default=2, type=int)
    parser.add_argument('--num_threads', default=64, type=int)
    parser.add_argument('--dataset_name', default='satellite', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.tensorflow_dir = args.tensorflow_data_dir
    args.train_directory = os.path.join(args.tensorflow_dir, 'train')
    args.validation_directory = os.path.join(args.tensorflow_dir, 'validation')
    # 输出数据保存目录
    args.output_directory = args.tensorflow_dir
    # 标签存放目录
    args.labels_file = os.path.join(args.tensorflow_dir, 'label.txt')
    # 判断标签目录是否存在
    if os.path.exists(args.labels_file) is False:
        logging.warning('Can\'t find label.txt. Now create it.')
        # 取出所有目录的标记
        all_entries = os.listdir(args.train_directory)
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w', encoding='utf-8') as f:
            for dirname in dirnames:
                f.write(dirname + '\n')
    # 执行main()函数
    main(args)
