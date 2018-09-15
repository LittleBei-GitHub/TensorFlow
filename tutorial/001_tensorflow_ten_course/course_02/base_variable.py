#coding=utf-8

import tensorflow as tf

a=tf.Variable([1, 2])
b=tf.Variable([3, 4])

add=tf.add(a, b)
sub=tf.subtract(a, b)

# 初始变量
state=tf.Variable(0, name='counter')
# 增加值
new_state=tf.add(state, 1)
# 更新操作，赋值
update=tf.assign(state, new_state)


init=tf.global_variables_initializer()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(add))
        print(sess.run(sub))

    with tf.Session() as sess:
        sess.run(init)
        for i in range(0, 10):
            value, _=sess.run(fetches=[state, update])
            print(value)
