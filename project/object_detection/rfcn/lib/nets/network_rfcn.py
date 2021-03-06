# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from layer_utils.proposal_target_layer_ohem import proposal_target_layer_ohem

from model.config import cfg

import pprint


class Network(object):
    def __init__(self, batch_size=1):
        self._feat_stride = [16, ]
        self._feat_compress = [1. / 16., ]
        self._batch_size = batch_size
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_image_summary(self, image, boxes, summarys_list):
        # add back mean
        image += cfg.PIXEL_MEANS
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)

        image_summary_op = tf.summary.image('ground_truth', image)
        summarys_list.append(image_summary_op)
        return image_summary_op

    def _add_act_summary(self, tensor, train_summaries):
        train_summaries.append(tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor))
        train_summaries.append(tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor)))

    def _add_score_summary(self, key, tensor, train_summaries):
        train_summaries.append(tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor))

    def _add_train_summary(self, var, train_summaries):
        train_summaries.append(tf.summary.histogram('TRAIN/' + var.op.name, var))

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name, mode='TRAIN'):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _proposal_target_layer_ohem(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer_ohem,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def build_network(self, sess, input_rois=None, roi_scores=None, is_training=True):
        raise NotImplementedError

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            self._losses['rpn_loss'] = rpn_loss_box + rpn_cross_entropy
            self._losses['class_loss'] = cross_entropy + loss_box
            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    def _add_rpn_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            self._losses['rpn_loss'] = rpn_loss_box + rpn_cross_entropy

            self._event_summaries.update(self._losses)

        return self._losses['rpn_loss']

    def _add_rfcn_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag) as scope:
            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box

            self._losses['rfcn_loss'] = cross_entropy + loss_box

            self._event_summaries.update(self._losses)

        return self._losses['rfcn_loss']

    def _smooth_l1_loss_vector(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0,
                               dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_rfcn_losses_ohem(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag) as scope:
            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            rfcn_cls_score = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label)

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box_vector = self._smooth_l1_loss_vector(bbox_pred, bbox_targets, bbox_inside_weights,
                                                          bbox_outside_weights)

            # ohem
            rois_boxes = self._proposal_targets['rois']
            loss_before_nms = rfcn_cls_score + loss_box_vector
            ohem_indexes = tf.image.non_max_suppression(rois_boxes[:, 1:5], loss_before_nms, cfg.TRAIN.OHEM_B,
                                                        cfg.TRAIN.OHEM_NMS_THRESH)
            rfcn_cls_score = tf.gather(rfcn_cls_score, ohem_indexes)
            loss_box_vector = tf.gather(rfcn_cls_score, ohem_indexes)

            cross_entropy = tf.reduce_mean(rfcn_cls_score)
            loss_box = tf.reduce_mean(loss_box_vector)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box

            self._losses['rfcn_loss'] = cross_entropy + loss_box
            self._losses['ohem_indexes_counts'] = ohem_indexes

            self._event_summaries.update(self._losses)

        return self._losses['rfcn_loss']

    def center_loss(self, features, label, alfa, nrof_classes=2):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
        nrof_features = features.get_shape()[1]
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label)
        diff = (1 - float(alfa)) * (centers_batch - features)
        centers = tf.scatter_sub(centers, label, diff)
        loss = tf.nn.l2_loss(features - centers_batch)
        return loss, centers

    def _add_losses_center(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            # center loss
            features = self._predictions['fc7']
            c_loss, _ = self.center_loss(features, label, cfg.CENTER_ALFA)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            self._losses['center_loss'] = c_loss

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + 0.01 * c_loss
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    def get_train_variables(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if scope in var.op.name:
                if 'rpn' in scope:
                    if 'refined' not in var.op.name:
                        vars.append(var)
                elif 'rfcn' in scope:
                    if 'rpn' not in var.op.name:
                        vars.append(var)
                else:
                    raise ValueError('illegle scope name')
        return vars

    def get_train_variables_stage3(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if scope in var.op.name:
                if 'rpn_conv' in var.op.name:
                    vars.append(var)
                elif 'rpn_cls_score' in var.op.name:
                    vars.append(var)
                elif 'rpn_bbox_pred' in var.op.name:
                    vars.append(var)
                else:
                    continue
        return vars

    def get_train_variables_stage4(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if scope in var.op.name:
                if 'refined' in var.op.name:
                    vars.append(var)
                else:
                    continue
        return vars

    def create_architecture(self, sess, mode, num_classes, scope, tag=None,
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), input_rois=None, roi_scores=None,
                            proposal_targets=None):
        # 图像
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        # 图像信息
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        # ground truth boxes
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        #
        self._tag = tag
        # 类别数量
        self._num_classes = num_classes
        # train/test
        self._mode = mode
        # 锚框范围
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
        # 锚框比例
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        # 每个点的锚框个数
        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d,
                        slim.conv2d_in_plane,
                        slim.conv2d_transpose,
                        slim.separable_conv2d,
                        slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            with tf.variable_scope(scope):
                rois, cls_prob, bbox_pred = self.build_network(sess, input_rois, roi_scores, training)

        if proposal_targets is not None:
            self._proposal_targets['rois'] = proposal_targets['rois']
            self._proposal_targets['labels'] = proposal_targets['labels']
            self._proposal_targets['bbox_targets'] = proposal_targets['bbox_targets']
            self._proposal_targets['bbox_inside_weights'] = proposal_targets['bbox_inside_weights']
            self._proposal_targets['bbox_outside_weights'] = proposal_targets['bbox_outside_weights']

        layers_to_output = {'rois': rois}
        layers_to_output.update(self._predictions)

        pprint.pprint(tf.trainable_variables())
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            if scope == 'rpn_network':
                self._add_rpn_losses()
                if cfg.TRAIN.OHEM:
                    self._add_rfcn_losses_ohem()
                else:
                    self._add_rfcn_losses()
            elif scope == 'rfcn_network':
                if cfg.TRAIN.OHEM:
                    self._add_rfcn_losses_ohem()
                else:
                    self._add_rfcn_losses()
            # self._add_losses_center()
            layers_to_output.update(self._losses)

        train_summaries = []
        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_image_summary(self._image, self._gt_boxes, train_summaries))
            for key, var in self._event_summaries.items():
                if 'ohem' in key:
                    loss_summary_op = tf.summary.scalar('LOSS/'+scope+'/'+key, tf.shape(var)[0])
                    val_summaries.append(loss_summary_op)
                    train_summaries.append(loss_summary_op)
                else:
                    loss_summary_op = tf.summary.scalar('LOSS/'+scope+'/'+key, var)
                    val_summaries.append(loss_summary_op)
                    train_summaries.append(loss_summary_op)
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var, train_summaries)
            for var in self._act_summaries:
                self._add_act_summary(var, train_summaries)
            for var in self._train_summaries:
                self._add_train_summary(var, train_summaries)

        train_summaries2 = []
        if scope == 'rfcn_network':
            for key in train_summaries:
                if 'rpn_network' not in key.op.name:
                    # print('#############'*3)
                    train_summaries2.append(key)
        if len(train_summaries2) > 0:
            self._summary_op = tf.summary.merge(train_summaries2)
        else:
            self._summary_op = tf.summary.merge(train_summaries)
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output, self._proposal_targets

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def get_summary_stage2(self, sess, blobs, network_rpn):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     network_rpn._image: blobs['data'], network_rpn._im_info: blobs['im_info'],
                     network_rpn._gt_boxes: blobs['gt_boxes']
                     }
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_rpn_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['rpn_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_rfcn_step_stage2(self, sess, blobs, train_op, network_rpn):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     network_rpn._image: blobs['data'], network_rpn._im_info: blobs['im_info'],
                     network_rpn._gt_boxes: blobs['gt_boxes']
                     }
        loss_cls, loss_box, loss, _ = sess.run([self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['rfcn_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return loss_cls, loss_box, loss

    def train_rfcn_step_stage4(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['rfcn_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_step_center(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, center_loss, loss, _ = sess.run(
            [self._losses["rpn_cross_entropy"],
             self._losses['rpn_loss_box'],
             self._losses['cross_entropy'],
             self._losses['loss_box'],
             self._losses['center_loss'],
             self._losses['total_loss'],
             train_op],
            feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, center_loss, loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['total_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_rpn_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_rpn, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['rpn_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_rpn, summary

    def train_rfcn_step_with_summary_stage2(self, sess, blobs, train_op, network_rpn):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     network_rpn._image: blobs['data'], network_rpn._im_info: blobs['im_info'],
                     network_rpn._gt_boxes: blobs['gt_boxes']
                     }
        loss_cls, loss_box, loss_rpn, summary, _ = sess.run([self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['rfcn_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return loss_cls, loss_box, loss_rpn, summary

    def train_rfcn_step_with_summary_stage4(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_rpn, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['rfcn_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_rpn, summary



    def train_step_with_summary_center(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, center_loss, loss, summary, _ = sess.run(
            [self._losses["rpn_cross_entropy"],
             self._losses['rpn_loss_box'],
             self._losses['cross_entropy'],
             self._losses['loss_box'],
             self._losses['center_loss'],
             self._losses['total_loss'],
             self._summary_op,
             train_op],
            feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, center_loss, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)

