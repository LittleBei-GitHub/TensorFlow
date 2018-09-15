# coding=utf8

'''
tf.nn.lrn(input, depth_radius, bias, alpha, beta, name)
Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).
第一个参数input：这个输入就是feature map了，既然是feature map，那么它就具有[batch,height,width,channels]这样的shape
第二个参数depth_radius：这个值需要自己指定，就是上述公式中的n/2
第三个参数bias：公式中的k
第四个参数alpha：公式中的α
第五个参数beta：公式中的β
返回值是新的feature map，它应该具有和原feature map相同的shape
'''