#coding=utf-8

## L2
    # tf.nn.l2_loss(t, name=None)
    # 这个函数的作用是利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半

## 交叉熵
    # 交叉熵是Loss函数的一种（也称为损失函数或代价函数），用于描述模型预测值与真实值的差距大小，
    # 常见的Loss函数就是均方平均差（Mean Squared Error），定义如下：
    # C=(y−a)^2/2
    # 平方差很好理解，预测值与真实值直接相减，为了避免得到负数取绝对值或者平方，再做平均就是均方平方差。
    # 注意这里预测值需要经过sigmoid激活函数，得到取值范围在0到1之间的预测值。
    # 平方差可以表达预测值与真实值的差异，但在分类问题种效果并不如交叉熵好。

    # 神经元的输出为a=σ(z)，这里z=∑jwjxj+b 。我们定义这个神经元的交叉熵代价函数为：
    # C=−1n∑x[ylna+(1−y)ln(1−a)]，
    # 这里n是训练数据的个数，这个加和覆盖了所有的训练输入x，y是期望输出。

##四个交叉熵函数
    # tf.nn.sigmoid_cross_entropy_with_logits
    # tf.nn.softmax_cross_entropy_with_logits
    # tf.nn.sparse_softmax_cross_entropy_with_logits
        # 它的第一个参数logits和前面一样，shape是[batch_size, num_classes]，
        # 而第二个参数labels以前也必须是[batch_size, num_classes]否则无法做Cross Entropy，
        # 这个函数改为限制更强的[batch_size]，而值必须是从0开始编码的int32或int64，而且值范围是[0, num_class)，
        # 如果我们从1开始编码或者步长大于1，会导致某些label值超过这个范围，代码会直接报错退出。
        # 这也很好理解，TensorFlow通过这样的限制才能知道用户传入的3、6或者9对应是哪个class，
        # 最后可以在内部高效实现类似的onehot encoding，这只是简化用户的输入而已，
        # 如果用户已经做了onehot encoding那可以直接使用不带“sparse”的softmax_cross_entropy_with_logits函数。
    # tf.nn.weighted_cross_entropy_with_logits
        # 是sigmoid_cross_entropy_with_logits的拓展版，输入参数和实现和后者差不多，
        # 可以多支持一个pos_weight参数，目的是可以增加或者减小正样本在算Cross Entropy时的Loss。
        # 实现原理很简单，在传统基于sigmoid的交叉熵算法上，正样本算出的值乘以某个系数接口