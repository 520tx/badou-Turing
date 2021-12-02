# [1] 读取 Cifar10 数据
# [2] 数据增强预处理
import os
import tensorflow as tf

num_classes = 10
# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000

# 定义一个空类，用于返回读取的 Cifar-10 数据
class CIFAR10Record(object):
    pass

# 定义一个读取 Cifar-10 数据的函数 read_cifar10(), 目的是读取目标文件的内容，并把数据分割，转化成 labels 和 images 的形式
def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1     # 如果是 Cifar10 数据集，则此处为 2
    result.height = 32
    result.width = 32
    result.depth = 3    # 因为 RGB 三通道，所以深度是 3
    image_bytes = result.height * result.width * result.depth # 图像样本总元素数量
    record_bytes = label_bytes + image_bytes # 因为每个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    # 使用tf.FixedLengthRecordReader()创建一个文件读取类，目的是读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue) # 使用该类的read()函数从文件队列里面读取文件
    record_bytes = tf.decode_raw(value, tf.uint8) # 读取到文件后，将读取到的文件内容从字符串形式解析为图像对应的像素数组

    # 因为该数组第一个元素是标签，所以使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转化成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下元素分割出来就是图片数据，这些数据在数据集里存储的形式是 depth * height * width
    # 要把这种格式转化成 [depth, height, width]，这一步将一维数据转化成三维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    # 将之前分割好的数据使用tf.transpose()函数转化成为高度信息、宽度信息、深度信息这样的顺序，转化数据排布方式为 (h, w, c)
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result # 返回值是已经把目标文件里的信息都读取出来

# 该函数对数据进行预处理 --> 对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i)for i in range(1, 6)] # 拼接地址
    file_queue = tf.train.string_input_producer(filenames) # 根据已经有的文件地址创建一个文件队列
    read_input = read_cifar10(file_queue) # 根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image = tf.cast(read_input.uint8image, tf.float32) # 将已经转化好的图片数据再次转化为float32的形式
    num_examples_per_epoch = num_examples_pre_epoch_for_train

    # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
    if distorted != None:
        # 1-将预处理好的图片进行剪切，使用tf.random_crop()函数
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 2-将剪切好的图片进行左右翻转，使用tf.image.random_filp_left_right()函数
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 3-将左右翻转好的图片进行随机高度调整，使用tf.image.random_brightness()函数
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 4-将高度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 5-进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        float_image.set_shape([24, 24, 3]) # 设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train. This will take a few minutes." % min_queue_examples)

        # 6-使用 tf.train.shuffle_batch() 函数随机产生一个batch的 image 和 label
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        return images_train, tf.reshape(labels_train, [batch_size])
    # 不对图像数据进行数据增强处理
    else:
        # 1-在这种情况下，使用函数 tf.image.resize_image_with_crop_or_pad() 对图片数据进行剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 2-剪切完成以后，直接进行图像标准化操作
        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)
        # 3-使用 batch() 函数代替 tf.train.shuffle_batch() 函数
        image_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                 batch_size=batch_size, num_threads=16,
                                                 capacity=min_queue_examples + 3 * batch_size)

        return image_test, tf.reshape(labels_test, [batch_size])