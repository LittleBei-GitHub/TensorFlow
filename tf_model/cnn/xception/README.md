# TensorFlow Xception
  解读Xception的源码，解读的所有源码全部来自下面的引用

  * Contents
    * xception.py: xception模型实现文件
    * xception_preprocessing.py: 图像预处理文件
    * xception_test.py: 测试xception模型
    * write_pb.py: 将训练好的模型转成pd形式
    * train_flowers.py: 训练flowers数据集的脚本
    * eval_flowers.py: 评估flowers数据集的脚本
    * dataset: flowers数据集的TFRecords格式版本
  * How to Run
    * 运行 python train_flowers.py 开始训练模型
    * 运行 tensorboard --logdir=log 可以可视化训练过程
  * Customization
    你可以简单地更改数据集文件和相应的名称（即名称为“flowers”的任何名称），以便将网络用于您自己的目的。
    重要的是，您应该能够获取自己的数据集的TFRecord文件以开始训练，因为数据管道依赖于TFRecord文件。
  * References
    * https://github.com/kwotsin/TensorFlow-Xception
    * https://github.com/kwotsin/create_tfrecords