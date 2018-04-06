#coding=utf-8

## Optimizer
    # Optimizer基类提供了用于计算损失函数梯度并将梯度应用到变量更新中取的方法。
    # TensorFlow实现了一系列经典的优化算法比如GradientDescent和Adagrad等，都继承了基类Optimizer。
    # 使用优化器的时候，无须实例化Optimizer这个基类，但要实例化他的某个子类。
    #
    # class tf.train.Optimizer
    # 这个类定义了一些API为计算图添加节点来训练模型。
    # 你无须直接使用这个类，但是要使用它的某个子类，比如：GradientDescentOptimizer，AdagradOptimizer，或者MomentumOptimizer

## 在使用梯度之前先处理梯度然后再去更新模型参数变量
    # 当调用minimize()时会有两个连续的步骤：计算题度，接着将其应用到变量上。
    # 如果你想在使用梯度之前对梯度做一些特别的处理，那么用下面这三个步骤：
    # 1.使用compute_gradients()计算梯度
    # 2.根据你的需求对计算出的梯度做一些处理
    # 3.使用apply_gradients()把预处理过的梯度拿来更新variables

    # eg:
    # # Create an optimizer.
    # opt = GradientDescentOptimizer(learning_rate=0.1)

    # # Compute the gradients for a list of variables.
    # grads_and_vars = opt.compute_gradients(loss, <list of variables>)
    #
    # # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # # need to the 'gradient' part, for example cap them, etc.
    # capped_grads_and_vars = [(MyCapper(gv[0]), gv[1])) for gv in grads_and_vars]

    # # Ask the optimizer to apply the capped gradients.
    # opt.apply_gradients(capped_grads_and_vars)

## tf.train.Optimizer.__init__(use_locking, name)
    # Create a new Optimizer.
    #
    # This must be called by the constructors of subclasses.
    #
    # Args:
    #
    # use_locking: Bool. If True apply use locks to prevent concurrent updates to variables.
    # name: A non-empty string. The name to use for accumulators created for the optimizer.
    # Raises:
    #
    # ValueError: if name is malformed.

## tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, name=None)
    # Args:
        # loss: A Tensor containing the value to minimize.
        # global_step: Optional Variable to increment by one after the variables have been updated.
        # var_list: Optional list of variables.Variable to update to minimize ‘loss’.
        #           Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
        # gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
        # name: Optional name for the returned operation.
    # Returns:
        # An Operation that updates the variables in ‘var_list’. If ‘global_step’ was not None, that operation also increments global_step.
    # Raises:
        # ValueError: if some of the variables are not variables.Variable objects.

## tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1)
    # Compute gradients of “loss” for the variables in “var_list”.
    # This is the first part of minimize().
    # It returns a list of (gradient, variable) pairs where “gradient” is the gradient for “variable”.
    # Note that “gradient” can be a Tensor, a IndexedSlices, or None if there is no gradient for the given variable.
    # Args:
        # loss: A Tensor containing the value to minimize.
        # var_list: Optional list of variables.Variable to update to minimize “loss”.
        #           Defaults to the list of variables collected in the graph under the key GraphKey.TRAINABLE_VARIABLES.
        # gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
    # Returns:
        # A list of (gradient, variable) pairs.
    # Raises:
        # TypeError: If var_list contains anything else than variables.Variable.
        # ValueError: If some arguments are invalid.

## tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)
    # Apply gradients to variables.
    # This is the second part of minimize(). It returns an Operation that applies gradients.
    # Args:
        # grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
        # global_step: Optional Variable to increment by one after the variables have been updated.
        # name: Optional name for the returned operation. Default to the name passed to the Optimizer constructor.
    # Returns:
        # An Operation that applies the specified gradients. If ‘global_step’ was not None, that operation also increments global_step.
    # Raises:
        # TypeError: if grads_and_vars is malformed.

## Gating Gradients
    # Both minimize() and compute_gradients() accept a gate_gradient argument that controls
    # the degree of parallelism during the application of the gradients.
    # The possible values are: GATE_NONE, GATE_OP, and GATE_GRAPH.
    # GATE_NONE: Compute and apply gradients in parallel. This provides the maximum parallelism in execution,
    #            at the cost of some non-reproducibility in the results.
    #            For example the two gradients of MatMul depend on the input values:
    #            With GATE_NONE one of the gradients could be applied to one of the inputs
    #            before the other gradient is computed resulting in non-reproducible results.
    # GATE_OP: For each Op, make sure all gradients are computed before they are used.
    #          This prevents race conditions for Ops that generate gradients for multiple inputs
    #          where the gradients depend on the inputs.
    # GATE_GRAPH: Make sure all gradients for all variables are computed before any one of them is used.
    #             This provides the least parallelism but can be useful
    #             if you want to process all gradients before applying any of them.


## Slots

    # Some optimizer subclasses, such as MomentumOptimizer and AdagradOptimizer allocate and manage
    # additional variables associated with the variables to train. These are called Slots.
    # Slots have names and you can ask the optimizer for the names of the slots that it uses.
    # Once you have a slot name you can ask the optimizer for the variable it created to hold the slot value.

    # This can be useful if you want to log debug a training algorithm, report stats about the slots, etc.

    # tf.train.Optimizer.get_slot_names()
        # Return a list of the names of slots created by the Optimizer.
        # See get_slot().
        # Returns:
        # A list of strings.
    # tf.train.Optimizer.get_slot(var, name)
        # Return a slot named “name” created for “var” by the Optimizer.
        #
        # Some Optimizer subclasses use additional variables.
        # For example Momentum and Adagrad use variables to accumulate updates.
        # This method gives access to these Variables if for some reason you need them.
        #
        # Use get_slot_names() to get the list of slot names created by the Optimizer.

        # Args:
            # var: A variable passed to minimize() or apply_gradients().
            # name: A string.
        # Returns:
            # The Variable for the slot if it was created, None otherwise.


## TensorFlow中的优化器子类
    # GradientDescentOptimizer
    # ProximalGradientDescentOptimizer
    # AdagradOptimizer
    # ProximalAdagradOptimizer
    # AdagradDAOptimizer
    # MomentumOptimizer
    # AdamOptimizer
    # FtrlOptimizer
    # RMSPropOptimize
    # AdadeltaOptimizer