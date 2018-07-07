# coding=utf-8

from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from train_flowers import get_split, load_batch
from xception import xception, xception_arg_scope
from tensorflow.python.framework import graph_util

import xception_preprocessing
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt


plt.style.use('ggplot')

# State your log directory where you can retrieve your model
# log目录，你可以检索你的模型
log_dir = './log'

# Create a new evaluation log directory to visualize the validation process
# 创建一个新的评估log目录，为了可视化验证过程
log_eval = './log_eval_test'

# State the dataset directory where the validation set is found
# 数据集目录
dataset_dir = './dataset'

# State the batch_size to evaluate each time, which can be a lot more than the training batch
# 批次大小
batch_size = 36

# State the number of epochs to evaluate
# 轮数
num_epochs = 1

# Get the latest checkpoint file
# 获取最近的检查点文件
checkpoint_file = tf.train.latest_checkpoint(log_dir)


def run():
    # Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)

    # Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        # 设置日志的级别，会将日志级别为INFO的打印出
        tf.logging.set_verbosity(tf.logging.INFO)

        # Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        # 获取验证数据集
        dataset = get_split('validation', dataset_dir)
        images, raw_images, labels = load_batch(dataset, batch_size=batch_size, is_training=False)

        # Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        # Now create the inference model but set is_training=False
        with slim.arg_scope(xception_arg_scope()):
            logits, end_points = xception(images, num_classes=dataset.num_classes, is_training=False)

        # get all the variables to restore from the checkpoint file and create the saver function to restore
        # 获取所有恢复的变量
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # 定义恢复函数
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Just define the metrics to track without the loss or whatsoever
        probabilities = end_points['Predictions']
        predictions = tf.argmax(probabilities, 1)

        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update)

        # Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)  # no apply_gradient method so manually increasing the global_step

        # Create a evaluation step function
        # 创建评估函数
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
            time_elapsed = time.time() - start_time

            # Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value,
                         time_elapsed)

            return accuracy_value

        # Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        # Get your supervisor
        # 创建监督器
        sv = tf.train.Supervisor(logdir=log_eval, summary_op=None, init_fn=restore_fn)

        # Now we are ready to run in one session
        with sv.managed_session() as sess:
            for step in xrange(int(num_batches_per_epoch * num_epochs)):
                # print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))

                # Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)

            # At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))

            # Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions, probabilities = sess.run([raw_images, labels, predictions, probabilities])
            for i in range(10):
                image, label, prediction, probability = raw_images[i], labels[i], predictions[i], probabilities[i]
                prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
                text = 'Prediction: %s \n Ground Truth: %s \n Probability: %s' % (
                    prediction_name, label_name, probability[prediction])
                img_plot = plt.imshow(image)

                # Set up the plot and hide axes
                plt.title(text)
                img_plot.axes.get_yaxis().set_ticks([])
                img_plot.axes.get_xaxis().set_ticks([])
                plt.show()

            logging.info(
                'Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    run()
