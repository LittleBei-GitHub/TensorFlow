# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



#从网址下载数据集存放到data_dir指定的目录中
def maybe_download_and_extract(data_dir):
    """下载并解压缩数据集 from Alex's website."""
    dest_directory = data_dir
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1] #'cifar-10-binary.tar.gz'
    filepath = os.path.join(dest_directory, filename)#'../CIFAR10_dataset\\cifar-10-binary.tar.gz'
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')#'../CIFAR10_dataset\\cifar-10-batches-bin'
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def get_distorted_train_batch(data_dir,batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

      Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

      Raises:
        ValueError: If no data_dir
      """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
    return images,labels

def get_undistorted_eval_batch(data_dir, batch_size, eval_data=True):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    Raises:
        ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,data_dir=data_dir,batch_size=batch_size)
    return images,labels