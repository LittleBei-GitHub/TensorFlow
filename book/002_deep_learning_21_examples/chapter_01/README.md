# 1. MNIST机器学习入门

## 1.1.1 简介

下载MNIST数据集，并打印一些基本信息：

```
python download.py
```

## 1.1.2 实验：将MNIST数据集保存为图片

```
python save_pic.py
```

## 1.1.3 图像标签的独热表示

打印MNIST数据集中图片的标签：
```
python label.py
```

## 1.2.1 Softmax 回归

```
python softmax_regression.py
```

## 1.2.2 两层卷积网络分类
```
python convolutional.py
```

## 可能出现的错误

下载数据集时可能出现网络问题，可以用下面两种方法中的一种解决：
1. 使用合适的代理
2.在MNIST的官方网站上下载文件train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz，并将它们存储在MNIST_data/文件夹中。
