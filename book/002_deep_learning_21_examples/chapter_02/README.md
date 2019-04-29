# 2. CIFAR-10与ImageNet图像识别

## 2.1.2 下载CIFAR-10 数据

```
python cifar10_download.py
```

## 2.1.3 TensorFlow 的数据读取机制

实验脚本：
```
python test.py
```

## 2.1.4 实验：将CIFAR-10 数据集保存为图片形式

```
python cifar10_extract.py
```

## 2.2.3 训练模型

```
python cifar10_train.py --train_dir cifar10_train/ --data_dir cifar10_data/
```

## 2.2.4 在TensorFlow 中查看训练进度
```
tensorboard --logdir cifar10_train/
```

## 2.2.5 测试模型效果
```
python cifar10_eval.py --data_dir cifar10_data/ --eval_dir cifar10_eval/ --checkpoint_dir cifar10_train/
```

使用TensorBoard查看性能验证情况：
```
tensorboard --logdir cifar10_eval/ --port 6007
```
