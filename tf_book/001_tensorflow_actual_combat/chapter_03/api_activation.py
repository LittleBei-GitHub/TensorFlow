#coding=utf-8

## 激活函数
    # 在神经网络中，我们有很多的非线性函数来作为激活函数，
    # 比如连续的平滑非线性函数（sigmoid，tanh和softplus），连续但不平滑的非线性函数（relu，relu6和relu_x）和随机正则化函数（dropout）。
    # 所有的激活函数都是单独应用在每个元素上面的，并且输出张量的维度和输入张量的维

## 分类：
    # 平滑非线性激活单元（sigmoid，tanh，elu，softplus，softsign）
    # 连续但不是处处可导（微）的激活单元（relu，relu6，crelu，relu_x）
    # 随机正则化激活单元（dropout）

## relu
    # tf.nn.relu(features, name = None)
    # 解释：这个函数的作用是计算激活函数relu，即max(features, 0)。
    # 输入参数：
        # features: 一个Tensor。数据类型必须是：float32，float64，int32，int64，uint8，int16，int8。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，数据类型和features相同。

## relu6
    # tf.nn.relu6(features, name = None)
    # 解释：这个函数的作用是计算激活函数relu6，即min(max(features, 0), 6)。
    # 输入参数：
        # features: 一个Tensor。数据类型必须是：float，double，int32，int64，uint8，int16或者int8。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，数据类型和features相同。

## crelu
    # tf.nn.crelu(features, name = None)
    # 解释：这个函数会倍增通道，一个是relu，一个是relu关于y轴对称的形状。
    # 输入参数：
        # features: 一个Tensor。数据类型必须是：float32，float64，int32，int64，uint8，int16，int8。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，数据类型和features相同，通道加倍。

## elu
    # tf.nn.elu(features, name = None)
    # 解释：x小于0时，y = a*(exp(x)-1)，x大于0时同relu。
        # features: 一个Tensor。数据类型必须是：float32，float64，int32，int64，uint8，int16，int8。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，数据类型和features相同。

## softplus
    # tf.nn.softplus(features, name = None)
    # 解释：这个函数的作用是计算激活函数softplus，即log( exp( features ) + 1)。
    # 输入参数：
        # features: 一个Tensor。数据类型必须是：float32，float64，int32，int64，uint8，int16或者int8。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，数据类型和features相同。

## softsign
    # tf.nn.softsign(features, name = None)
    # 解释：这个函数的作用是计算激活函数softsign，即features / (abs(features) + 1)。
    # 输入参数：
        # features: 一个Tensor。数据类型必须是：float32，float64，int32，int64，uint8，int16或者int8。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，数据类型和features相同。

## sigmoid
    # tf.sigmoid(x, name = None)
    # 解释：这个函数的作用是计算 x 的 sigmoid 函数。具体计算公式为 y = 1 / (1 + exp(-x))。
    # 输入参数：
        # x: 一个Tensor。数据类型必须是float，double，int32，complex64，int64或者qint32。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，如果 x.dtype != qint32 ，那么返回的数据类型和x相同，否则返回的数据类型是 quint8 。

## tanh
    # tf.tanh(x, name = None)
    # 解释：这个函数的作用是计算 x 的 tanh 函数。具体计算公式为 ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )。
        # x: 一个Tensor。数据类型必须是float，double，int32，complex64，int64或者qint32。
        # name: （可选）为这个操作取一个名字。
    # 输出参数：一个Tensor，如果 x.dtype != qint32 ，那么返回的数据类型和x相同，否则返回的数据类型是 quint8 。