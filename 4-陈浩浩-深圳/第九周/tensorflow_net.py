import tensorflow as tf
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape, x_test.shape)


nClass = tf.placeholder(tf.int32)
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.int32, shape=[None])
y_hot = tf.one_hot(indices=y, depth=nClass, on_value=1.0, off_value=0.0, axis=-1)

# 网络层
w1 = tf.Variable(initial_value=tf.random_normal(shape=[784, 100], stddev=0.1), name="weight1")
b1 = tf.Variable(initial_value=tf.constant(0.1, shape=(1, 100)), name="bias1")
h1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = tf.Variable(initial_value=tf.random_normal(shape=[100, 20], stddev=0.1), name="weight2")
b2 = tf.Variable(initial_value=tf.constant(0.1, shape=(1, 20)), name="bias2")
h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)

w3 = tf.Variable(initial_value=tf.random_normal(shape=[20, 10], stddev=0.1), name="weight2")
b3 = tf.Variable(initial_value=tf.constant(0.1, shape=(1, 10)), name="bias2")
h3 = tf.matmul(h2, w3)+b3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_hot, logits=h3))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)   # 梯度下降法，学习率为0.01
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)   # 梯度下降法，学习率为0.01


# 测试
prediction = tf.equal(tf.argmax(h3, axis=1), tf.argmax(y_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))

with tf.Session() as sess:
    epochs = 300
    batch_size = 8
    n = x_train.shape[0]
    # n = 10000
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        epoch_loss = 0.0
        iter_count = 0
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            batch_data = x_train[i:j, :, :].reshape(-1, 28 * 28)
            batch_label = y_train[i:j]
            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_data, y: batch_label, nClass: 10})
            # if epoch % 40 == 0:
            #     train_accuracy = accuracy.eval(feed_dict={
            #         x: batch_data, y: batch_label, nClass: 10})
            #     print('step %d, training accuracy %g' % (i, train_accuracy))
            epoch_loss += loss
            iter_count += 1

        print("[Epoch {}/{}], train loss: {}".format(epoch, epochs, epoch_loss/iter_count))



    # 测试：
    acc = 0
    iter_count = 0
    n = x_test.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch_data = x_test[i:j, :, :].reshape(-1, 28 * 28)
        batch_label = y_test[i:j]
        acc += sess.run(accuracy, feed_dict={x:batch_data, y:batch_label, nClass:10})
        iter_count += 1

    print('Test accuracy {:.4f}%'.format(100*acc/iter_count))    # Test accuracy 92.3077%

