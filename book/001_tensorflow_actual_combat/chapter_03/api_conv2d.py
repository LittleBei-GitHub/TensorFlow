#coding=utf-8

##tf.nn.conv2d是TensorFlow里面实现卷积的函数，参考文档对它的介绍并不是很详细，实际上这是搭建卷积神经网络比较核心的一个方法，非常重要

##tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    # 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
    #
    # 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
    # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # 第四个参数padding：string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同的卷积方式（后面会介绍）
    # 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true结果返回一个Tensor，这个输出，就是我们常说的feature map

## 输入张量的shape，卷积滤波器核的shape，输出张量的shape如何计算？
    # 输入数据体的尺寸为W1×H1×D1
        # 4个超参数：
            # 滤波器的数量K
            # 滤波器的控件尺寸F
            # 步长S
            # 零填充数量P
    # 输出数据体的尺寸为W2×H2×D2
        # W2=（W1−F+2P）/S+1
        # H2=(H1−F+2P)/S+1(宽度和高度的计算方法相同)
        # D2=K
    # 由于卷积运算在一个批次的样本之间是独立处理每个样本的，所以上面公式是计算单个样本被卷积以后的尺寸的。
    # 输入batchsize个张量，输出还是batchsize个张量，发生变化的是每个输入张量的尺寸。

    # 在使用tf.nn.conv2d设计卷积层的时候，我们首先要做的是确定权重weights和baises的shape。
    # 上述公式中的超参数K和F确定了权重weights（也就是filter）的shape：weights的shape[F,F,D1,K]
    # 上述公式中的超参数K确定了偏置biases的shape=[K]
    # 调节上述公式中的超参数F,S,P可以控制滤波输出的张量的尺寸。

##如果你想让输入张量的空间尺寸与输出张量的空间尺寸保持一致，那么padding=“SAME”，strides=[1,1,1,1]
##设置padding=“SAME”，公式里面的零填充数量P是TensorFlow自动推算的。
# P=((W2−1)∗S−W1+F)/2
    # 比如：
    # input：B1=10，W1=28，H1=28，D1=3
    # filter：F=3，S=1，P=1，K=10
    # output：
    # B2=B1=10（batch size保持不变）
    # W2=（28-2+2*P）/1 + 1 = 28
    # H2=（28-2+2*P）/1 + 1 = 28
    # D2 = K = 10
    
    # 比如：
    # input：B1=10，W1=28，H1=28，D1=3
    # filter：F=5，S=1，P=2，K=10
    # output：
    # B2=B1=10（batch size保持不变）
    # W2=（28-5+2*P）/1 + 1 = 28
    # H2=（28-5+2*P）/1 + 1 = 28
    # D2 = K = 10

##如果padding=“VALID”，那么输出数据体的尺寸怎么计算呢？

    # 输出数据体的尺寸W1×H1×D1
        # 4个超参数：
            # 滤波器的数量K
            # 滤波器的控件尺寸F
            # 步长S
            # 零填充数量P
        # 输出数据体的尺寸为W2×H2×D2，其中：
        # W2=（W1−F+2P）/S+1
        # H2=(H1−F+2P)/S+1(宽度和高度的计算方法相同)
        # D2=K

# 设置padding=“VALID”，公式里面的零填充数量P=0，也就是不填充
    # 比如：
    # input：B1=10，W1=28，H1=28，D1=3
    # filter：F=3，S=1，P=0，K=10
    # output：
    # B2=B1=10（batch size保持不变）
    # W2=（28-3+2*0）/1 + 1 = 26
    # H2=（28-3+2*0）/1 + 1 = 26
    # D2 = K = 10

    # 比如：
    # input：B1=10，W1=28，H1=28，D1=3
    # filter：F=5，S=1，P=0，K=10
    # output：
    # B2=B1=10（batch size保持不变）
    # W2=（28-5+2*0）/1 + 1 = 24
    # H2=（28-5+2*0）/1 + 1 = 24
    # D2 = K = 10
    # 设置了padding=“VALID”，卷积以后输出张量的空间尺寸会变小F-1圈。
    # 由于此处输入空间尺寸为偶数，滤波器核为奇数，所以不要把S设为2，否则不能整除，所以S=1（S=3步子有点大了，也不好）。如果想大幅度缩减尺寸，后面接pool层。