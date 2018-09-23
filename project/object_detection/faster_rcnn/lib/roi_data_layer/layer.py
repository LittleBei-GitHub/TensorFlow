# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """
            Randomly permute the training roidb.
            随机排列
        """
        self._perm = np.random.permutation(np.arange(len(self._roidb)))  # 随机排列
        self._cur = 0  # 游标

    def _get_next_minibatch_inds(self):
        """
            Return the roidb indices for the next minibatch.
            返回下一个小批次的roi索引
        """
        
        if cfg.TRAIN.HAS_RPN:
            # 如果大于图片的总数量（相当于做完一轮训练），则重新随机排列
            if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
                self._shuffle_roidb_inds()
            # 取出小批次索引
            db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
            # 更新游标
            self._cur += cfg.TRAIN.IMS_PER_BATCH
        else:
            # sample images
            db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
            i = 0
            while (i < cfg.TRAIN.IMS_PER_BATCH):
                ind = self._perm[self._cur]
                # 判断box数量
                num_objs = self._roidb[ind]['boxes'].shape[0]
                if num_objs != 0:
                    db_inds[i] = ind
                    i += 1
                # 更新游标
                self._cur += 1
                if self._cur >= len(self._roidb):
                    self._shuffle_roidb_inds()
        return db_inds

    def _get_next_minibatch(self):
        """
            Return the blobs to be used for the next minibatch.
            If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
            separate process and made available through self._blob_queue.
            返回用于下一个批次的blob
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)
            
    def forward(self):
        """
            Get blobs and copy them into this layer's top blob vector.
            获取blob
        """
        blobs = self._get_next_minibatch()
        return blobs
