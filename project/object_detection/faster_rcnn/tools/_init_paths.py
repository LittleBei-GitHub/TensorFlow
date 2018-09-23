# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys


def add_path(path):
    """
        添加路径到系统路径中
    """
    if path not in sys.path:
        sys.path.insert(0, path)


# 当前文件所在路径
this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
# 添加lib包到系统路径
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)


lib_path = osp.join(this_dir, 'mftracker')
add_path(lib_path)
