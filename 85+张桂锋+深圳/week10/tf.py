import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

'''
#使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
#print (x_data)
noise=np.random.normal(0,0.02,x_data.shape)
#print(x_data.shape)
#print(noise)
y_data=np.square(x_data)+noise
#print(y_data)
#定义两个placeholder存放输入数据
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))    #加入偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1)   #加入激活函数

#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))  #加入偏置项
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)   #加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
'''
#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#定义两个placeholder存放输入数据
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#定义神经网络中间层

Weights_L1=tf.Variable(tf.truncated_normal([784,200],stddev = 0.1))
biases_L1=tf.Variable(tf.zeros(200))    #加入偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
#print(Wx_plus_b_L1.shape)
L1=tf.nn.relu(Wx_plus_b_L1)   #加入激活函数
#print(L1.shape)
#定义神经网络输出层

Weights_L2=tf.Variable(tf.truncated_normal([200,10],stddev = 0.1))
biases_L2=tf.Variable(tf.zeros(10))  #加入偏置项
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.softmax(Wx_plus_b_L2)   #加入激活函数
#print(prediction.shape)
#print(y.shape)

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))

# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
# tf.argmax(input, axis=None, name=None, dimension=None)此函数是对矩阵按行或列计算最大值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
# tf.cast(x, dtype, name=None) ,把x转化为dtype型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size
print(n_batch)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):#训练21次
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+"Test Accuracy " + str(acc))


