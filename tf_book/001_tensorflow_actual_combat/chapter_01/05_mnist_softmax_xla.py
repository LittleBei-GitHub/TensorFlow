# coding=utf-8

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

FLAGS = None


def main(_):
    ## 加载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    ## 创建模型
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    ## 定义优化器和损失
    y_ = tf.placeholder(tf.int64, [None])

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    ## tf.ConfigProto()用来对session进行参数配置
    config = tf.ConfigProto()
    jit_level = 0
    if FLAGS.xla:
        # 打开 XLA（加速线性代数） JIT 编译器
        # XLA 使用 JIT 编译技术来分析用户在运行时创建的TensorFlow图，
        # 专门用于实际运行时的维度和类型，它将多个op融合在一起并为它们
        # 形成高效的本地机器代码--能用于CPU、GPU和自定义加速器（如谷歌TPU）
        #
        # XLA 是编译调试的秘密武器，它能帮助TensorFlow自动优化原始op的组合。
        # 有了XLA的增强，通过在运行时的过程中分析图、融合多个op并为融合子图
        # 生成有效的机器代码，TensorFlow能在保留其灵活性的同时而不牺牲运行时
        # 的性能。
        #
        # 总之，XLA帮助TensorFlow保持其灵活性，同时消除性能问题。
        jit_level = tf.OptimizerOptions.ON_1

    config.graph_options.optimizer_options.global_jit_level = jit_level
    # tf.RunMetadata()定义TensorFlow运行的元信息，这样可以记录训练时运行时间和内存占用等方面的信息
    run_metadata = tf.RunMetadata()
    sess = tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)
    ## 开始训练
    train_loops = 1000
    for i in range(train_loops):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        if i == train_loops - 1:
            # tf.RunOptions定义TensorFlow运行选项，其中设置trace_level为FULL_TRACE
            sess.run(train_step,
                     feed_dict={x: batch_xs, y_: batch_ys},
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata)
            # 创建timeline,并导入到json中，用chrome://tracing/查看
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())
        else:
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    ## 模型测试
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()


if __name__ == '__main__':
    ## python 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--xla', type=bool, default=True, help='Turn xla via JIT on')
    # FLAGS:将参数解析到FLAGS中
    # unparsed:没有解析的参数
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
