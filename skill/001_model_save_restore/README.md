# TensorFlow模型的保存和加载方式

TensorFlow保存和加载模型的方式总共有三种

## 保存为checkpoint形式
* checkpoint适用于原图代码存在的形式，此时我们可以忽略meta数据，加载模型只是为了恢复训练好的参数
* [checkpoint方式代码](https://github.com/lovejing0306/TensorFlow/blob/master/tf_skill/001_model_save_restore/model_save_restore_checkpoint.py)

## 保存为checkpoint和meta形式
* checkpoint和mete形式，此种方式不需要保留原图代码，优势在与容易部署，<br>在使用时可以直接通过变量name属性的命名来访问变量
* [checkpoint和meta方式代码](https://github.com/lovejing0306/TensorFlow/blob/master/tf_skill/001_model_save_restore/model_save_restore_checkpoint_meta.py)

## 保存为pb形式
* pb形式，此种方式也不需要保留原图代码，与checkpoint和mete形式相比，<br>除了具有容易部署、可直接通过变量name属性名访问变量外，还有生成的保存文件简洁的优势，<br>其只会生成一个pb文件
* pb形式有两种保存方式<br>[pb方式一代码](https://github.com/lovejing0306/TensorFlow/blob/master/tf_skill/001_model_save_restore/model_save_restore_pb_1.py)<br>[pb方式二代码](https://github.com/lovejing0306/TensorFlow/blob/master/tf_skill/001_model_save_restore/model_save_restore_pb_2.py)