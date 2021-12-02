import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)

y_data=np.square(x_data)+noise

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
wx_plus_b=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(wx_plus_b)



Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
L2=wx_plus_b_L2


loss=tf.reduce_mean(tf.square(y-L2))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction=sess.run(L2,feed_dict={x:x_data})
    writer = tf.summary.FileWriter('logs', tf.get_default_graph())
    writer.close()

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction,'r-',lw=10)
    plt.show()






