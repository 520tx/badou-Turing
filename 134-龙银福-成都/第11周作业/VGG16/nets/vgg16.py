# -------------------------------------------------------------#
#        vgg16的网络部分
# -------------------------------------------------------------#
import tensorflow as tf

slim = tf.contrib.slim # 创建slim对象

# 建立vgg_16的网络
def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5,
           spatial_squeeze=True, scope='vgg_16'):
    with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs]):
        # [1] 第一个卷积块(conv-1)：2个卷积层，输出的特征层为64(卷积核数量)，两次卷积均使用3x3的卷积核
        # 输入(224, 224, 3)，输出(224, 224, 64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 第一个最大池化层：核尺寸2x2
        # 输入(224, 224, 64)，输出net为(112, 112, 64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # [2] 第二个卷积块(conv2)：2个卷积层，输出的特征层为128(卷积核数量)，两次卷积均使用3x3卷积核
        # 输入(112, 112, 64)，输出net为(112, 112, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 第二个最大池化层：核尺寸2x2
        # 输入(112, 112, 128)，输出net为(56, 56, 128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # [3] 第三个卷积块(conv3)：3个卷积层，输出的特征层为256(卷积核数量)，三次卷积均使用3x3卷积核
        # 输入(56, 56, 128)，输出net为(56, 56, 256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 第三个最大池化层：核尺寸2x2
        # 输入(56, 56, 256)，输出net为(28, 28, 256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # [4] 第四个卷积块(conv3)：3个卷积层，输出的特征层为512(卷积核数量)，三次卷积均使用3x3卷积核
        # 输入(28, 28, 256)，输出net为(28, 28, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 第四个最大池化层：核尺寸2x2
        # 输入(28, 28, 512)，输出net为(14, 14, 512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # [5] 第五个卷积块(conv3)：3个卷积层，输出的特征层为512(卷积核数量)，三次卷积均使用3x3卷积核
        # 输入(14, 14, 512)，输出net为(14, 14, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 第五个最大池化层：核尺寸2x2
        # 输入(14, 14, 512)，输出net为(7, 7, 512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # [6] 第六个block：1个卷积层，输出特征层为4096(卷积核数量)，卷积核大小7x7
        # 利用卷积的方式模拟全连接层，效果等同(规则：将卷积核大小设置为输入的空间大小)
        # 输入(7, 7, 512)，输出net为(1, 1, 4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        # dropout防止过拟合
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        # [7] 第七个block：1个卷积层，输出特征层为4096(卷积层数量)，卷积核大小1x1
        # 利用卷积的方式模拟全连接层，效果等同
        # 输入(1, 1, 4096)，输出net为(1, 1, 4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # dropout防止过拟合
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # [8] 第八个block：1个卷积层，输出特征层为1000(卷积层数量)，卷积核大小1x1
        # 利用卷积的方式模拟全连接层，效果等同
        # 输入(1, 1, 4096)，输出net为(1, 1, 1000)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        # tf.squeeze()函数返回一个张量，这个张量是将原始net中所有维度为1的那些维都删掉的结果
        # axis=[]可以用来指定要删掉的为1的维度，此处要注意指定的维度必须确保其是1，否则会报错
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net