# RCNN

使用TF-Learn实现R-CNN模型

## 环境
TensorFlow版本：0.7.1

Python版本：2.7

scikit-learn开发包

tflearn开发包，注意tflearn开发包的版本要和TensorFlow的版本对应

selective search算法包

## 训练输入
这里使用17 flowers作为训练数据集，
数据集的下载[地址](https://github.com/ck196/tensorflow-alexnet)
数据文件为`17flowers.tar.gz`

## 项目结构
train_alexnet.py脚本使用具有train_list.txt文件的17flowers图像文件夹来执行Alexnet的预训练。这是需要先运行的文件。

生成model_save.model文件之后，你可以运行fine_tune_RCNN.py脚本，它使用2flowers数据集以及在svm_train下的两个txt文件对模型进行微调。

完成后，将创建名为fine_tune_model_save.model的文件作为微调模型结果。

最后，你可以运行RCNN_output.py文件来对图像进行分类。

在文件的主要部分中，更改图像名称以在不同图像上进行测试。现在，我只用两种花来调整图像，即三色堇和郁金香。

## 项目流程
* 预训练Alexnet模型
* 生成用于finetuning的数据集
* 模型finetuning
* 生成用于训练svm的数据集
* 训练svm分类器
* 训练bounding boxes（该模型中未给出，包括非极大值抑制）