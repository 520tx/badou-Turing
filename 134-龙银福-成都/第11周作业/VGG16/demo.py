from nets import vgg16
import tensorflow as tf
import utils

if __name__ == '__main__':
    # 读取图片
    img1 = utils.load_image("./test_data/table.jpg")

    inputs = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
    # 对输入的图片进行resize，使其shape满足(-1, 224, 224, 3)
    resized_img = utils.resize_image(inputs, (224, 224))

    # 建立网络模型结构，最后结果进行softmax预测
    prediction = vgg16.vgg_16(resized_img)
    pro = tf.nn.softmax(prediction)

    with tf.compat.v1.Session() as sess:
        # 所有变量的初始化
        sess.run(tf.compat.v1.global_variables_initializer())
        # 载入模型
        ckpt_filename = './model/vgg_16.ckpt'
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_filename)
        # softmax的预测
        pre = sess.run(pro, feed_dict={inputs:img1})
        # 打印预测结果
        print("result: ")
        # pre.shape=(1, 100)  二维 --> [[1,1,...,1]]
        # pre[0].shape=(100,) 一维 --> [1,1,...,1]
        utils.print_prob(pre[0], './synset.txt')