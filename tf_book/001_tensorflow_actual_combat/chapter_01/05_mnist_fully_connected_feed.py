# coding=utf-8

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# 保存模型的基本参数
FLAGS = None


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS), name='images')
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def run_training():
    """Train MNIST for a number of steps."""
    # 获取mnist的训练集、验证集和测试集
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # 创建默认的计算图
    with tf.Graph().as_default():
        # 生成输入数据images和labels的占位符
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # 模型输出
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # 模型损失
        loss = mnist.loss(logits, labels_placeholder)
        # 训练操作
        train_op = mnist.training(loss, FLAGS.learning_rate)
        # 评估操作
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # 合并所有的summary
        summary = tf.summary.merge_all()

        # 所有变量初始化操作
        init = tf.global_variables_initializer()

        # 创建保存checkpoints的saver
        saver = tf.train.Saver() # Saver可以选择要保存的参数

        # 开启session
        sess = tf.Session()

        # 保存计算图
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)


        # 初始化变量
        sess.run(init)

        # 开启训练的循环操作
        for step in xrange(FLAGS.max_steps):
            # 记录开始时间
            start_time = time.time()

            # 获取feed_dict
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
            # 获取损失
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # 计算花费的时间
            duration = time.time() - start_time

            # 每隔100步打印训练信息，保存summary
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step) # 添加summary
                summary_writer.flush() # 缓冲summary

            # 每隔1000步保存checkpoint，并对模型做出评估
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # 在训练集上评估
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)  # 传入eval_correct操作
                # 在验证集上评估
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                # 在测试集上评估
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    ## 开始训练
    run_training()


if __name__ == '__main__':
    # 定义一个参数解析，统一模型中用到的参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../logs',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    # 将参数解析到FLAGS中
    FLAGS, unparsed = parser.parse_known_args()
    # 开始执行程序
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
