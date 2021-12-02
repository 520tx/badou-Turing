import tensorflow as tf
# import tensorflow.compat.v1 as tf
import Cifar10_data
import time
import math
import numpy as np

# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape, stddev, w):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))   # 截断的正态分布，超过两倍标准偏差的被舍弃掉
    if w is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return var


def conv2d(x, w, strides):
    return tf.nn.conv2d(x, w, strides, padding='SAME')
    #图片X:shape(batch,h,w,channel); 卷积核w:(h,w,in_channel, out_channel); strides:[1, stride, stride, 1];
    # padding: 'SAME'考虑边界，填充0，'VALID'不填充边界


# 1. 参数设置
batch_size = 32
cifar_data_dir = r"D:\BaiduNetdiskDownload\【10】框架&CNN\代码\Cifar\cifar_data\cifar-10-batches-bin"
# cifar_data_dir = r"./cifar_data/cifar-10-batches-bin"  # 二进制文件格式存放
num_examples_for_eval = 10000   # 10000张测试图片
num_examples_for_train = 50000  # 50000 张训练图片；   （一个epoch训练：50000/8 个batch）
max_steps =  int(math.ceil(num_examples_for_train)/batch_size)*300   # 训练300个epoch


# 2. 数据加载
# from keras.datasets import cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
images_train, labels_train = Cifar10_data.inputs(data_dir=cifar_data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=cifar_data_dir, batch_size=batch_size, distorted=None)
print(images_train.shape, labels_train.shape, images_test.shape, labels_test.shape)


# 3. 网络搭建
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 24, 24, 3])  # cifar10: 32*32*3, 前处理crop到24*24*3
y_ = tf.placeholder(dtype=tf.int32, shape=[batch_size])

# 卷积层1
w_conv1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w=0)  # 卷积核输入channel是3，输出channel是64
b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]))  # 初始化为0
h_conv1 = tf.nn.relu(conv2d(x, w_conv1, strides=[1, 1, 1, 1])+b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积层2
w_conv2 = variable_with_weight_loss([5,5,64,64], stddev=5e-2, w=0)
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1])+b_conv2)
h_pool2 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# reshape
x_flat = tf.reshape(h_pool2, shape=[batch_size, -1])  # h_pool2:shape(7,7,64),需要reshape，方便全连接
dim = x_flat.get_shape()[1].value

# 全连接层1
w_fc1 = variable_with_weight_loss([dim, 384], stddev=0.04, w=0.004)
b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
h_fc1 = tf.nn.relu(tf.matmul(x_flat, w_fc1) + b_fc1)

# 全连接层2
w_fc2 = variable_with_weight_loss([384, 192], stddev=0.04, w=0.004)
b_fc2 = tf.Variable(tf.constant(0.1, shape=[192]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

# dropout
keep_prob = tf.placeholder(tf.float32)  # train是设置丢弃概率；eval时可以设置为1
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 全连接层3,预测输出
w_fc3 = variable_with_weight_loss([192, 10], stddev=1 / 192.0, w=0.0)
b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_output = tf.add(tf.matmul(h_fc2_drop, w_fc3), b_fc3)

# 损失函数
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_logits(labels=y_, logits=y_output))
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_,tf.int64), logits=y_output)
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

# 训练优化器
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 测试
top_k_op = tf.nn.in_top_k(y_output, y_, 1)  # 计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
# correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))  # cast数据转换

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化定义的变量
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    for step in range(max_steps):
            t1 = time.time()
            image_batch, label_batch = sess.run([images_train, labels_train])   #返回一个batch的数据
            # print(step, image_batch.shape, label_batch)
            loss_value, _ = sess.run([loss, train_step], feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
            duration = time.time() - t1
            if step % 100 == 0:
                # train_accuracy = accuracy.eval(feed_dict={
                #     x: image_batch, y_: label_batch, keep_prob: 1.0})
                print("[Step {}/{}] loss={:.4f}, {:.1f} images/sec, {} sec/batch".format(
                    step, max_steps, loss_value, batch_size/duration, duration))

    # 测试阶段
    num_batch = int(math.ceil(num_examples_for_eval/batch_size))  # 有多少个batch
    true_count = 0
    total_sample_count=num_batch * batch_size   # 若最后一个batch，图片不够一个batch_size时，注意是如何处理的？
    for i in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch, keep_prob: 1})
        true_count += np.sum(predictions)
    print("Test accuracy: {:.4f}%".format(100*true_count/total_sample_count))