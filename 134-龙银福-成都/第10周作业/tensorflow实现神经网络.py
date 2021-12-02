import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test_tensorflow():
    # 1-生成数据：numpy生成200个随机点
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] # 生成等差数列, 并添加一个维度, shape=[200, 1]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise
    x = tf.compat.v1.placeholder(tf.float32, [None, 1]) # 定义两个placeholder存放输入数据
    y = tf.compat.v1.placeholder(tf.float32, [None, 1])
    # 2-定义神经网络
    # 中间层
    weights_L1 = tf.Variable(tf.random.normal([1, 10]))
    biases_L1 = tf.Variable(tf.zeros([1, 10])) # 偏置
    Wx_plus_b_L1 = tf.matmul(x, weights_L1) + biases_L1 # op 运算
    L1 = tf.nn.tanh(Wx_plus_b_L1) # 激活
    # 输出层
    weights_L2 = tf.Variable(tf.random.normal([10, 1]))
    biases_L2 = tf.Variable(tf.zeros([1, 1]))
    Wx_plus_b_L2 = tf.matmul(L1, weights_L2) + biases_L2
    prediction = tf.nn.tanh(Wx_plus_b_L2)
    # 3-定义损失函数(均方差函数)和反向传播算法（使用梯度下降算法训练）
    loss = tf.reduce_mean(tf.square(y - prediction))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
    # 4-训练和推理
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer()) # 变量初始化
        for i in range(2000): # 训练2000次
            sess.run(train_step, feed_dict={x: x_data, y: y_data})
        prediction_value = sess.run(prediction, feed_dict={x:x_data})
        # 画图
        plt.figure()
        plt.scatter(x_data, y_data) # 画散点图, 散点是真实值
        plt.plot(x_data, prediction_value, 'r-', lw=5) # 预测值：曲线
        plt.show()


if __name__ == '__main__':
    [6]
    # tensorflow实现神经网络
    test_tensorflow()

    '''
    [5]
    # tensorboard --> 图可视化工具
    # 1-构建图graph
    a = tf.constant([10., 20., 40.], name='a')
    b = tf.Variable(tf.random_uniform([3]), name='b')
    output = tf.add_n([a, b], name='add')
    # 2-生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
    writer = tf.compat.v1.summary.FileWriter('logs', tf.get_default_graph())
    writer.close()
    # 3-启动 tensorboard 服务(在命令行启动)
    # tensorboard --logdir "E:\badou-Turing\134-龙银福-成都\第10周作业\logs" --host=127.0.0.1
    # 4-启动 tensorboard 服务后，复制地址并在本地浏览器中打开
    '''

    '''
    [4]
    # tensorflow feed --> 填充模板, 赋值
    input1 = tf.compat.v1.placeholder(tf.float32) # 定义一个模板(占位符), 后续利用feed_dict的字典结构给placeholder填充具体的内容
    input2 = tf.compat.v1.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.compat.v1.Session() as sess:
        print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
    '''

    '''
    [3]
    # tensorflow fetch --> 取出
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.multiply(input1, intermed)
    with tf.compat.v1.Session() as sess:
        result = sess.run([mul, intermed])
        print(result)
    '''

    '''
    [2]
    # tensorflow variable
    # 1-构建图graph
    state = tf.Variable(0, name="counter")
    init_op = tf.compat.v1.global_variables_initializer() # variable必须经过 init op 初始化，必须添加一个 init op 到图中
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.compat.v1.assign(state, new_value) # 把 new_value 的值赋给 state，state 的值必须是 Variable
    # 2-启动图，运行 op
    with tf.compat.v1.Session() as sess:
        sess.run(init_op) # 运行 init op, 初始化 Variable 的值
        print("state: ", sess.run(state))
        # 运行 op, 更新并打印 state
        for _ in range(5):
            print("update: ", sess.run(update))
            print("state: ", sess.run(state))
    '''

    '''
    [1]
    # tensorflow graph
    # 1-构建图(使用默认图 -- default graph)
    matrix1 = tf.constant([[3., 3.]]) # 常量 op
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    # 2-启动一个会话 --> 启动默认图
    sess = tf.compat.v1.Session()
    result = sess.run(product)
    print(result)
    sess.close()
    # 2-启动一个会话(启动默认图)，Session 对象使用完成后需要关闭以释放资源，
    # 除了显式调用 close() 关闭外，也可以使用 with 代码块来自动完成关闭动作
    with tf.compat.v1.Session() as sess:
        result = sess.run(product)
        print(result)
    '''