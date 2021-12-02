import os
import tensorflow as tf
num_classes = 10
num_examples_per_epoch_for_train = 500
num_examples_per_epoch_for_eval = 100
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class CIFA10Record(object):
    pass
def read_cifar10(file_queue):
    result = CIFA10Record()
    label_bytes = 1
    result.height = 32
    result.width =32
    result.depth = 3
    image_bytes = result.height*result.width*result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value,tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    depth_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),[result.depth,result.height,result.width])
    result.uint8image = tf.transpose(depth_major,[1,2,0])
    return result
def inputs(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir,"data_batch_%d.bin"%i) for i in range(1,6)]
    print(filenames)
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)
    reshape_image = tf.cast(read_input.uint8image,tf.float32)
    num_examples_per_epoch = num_examples_per_epoch_for_train
    if distorted != None:
        cropped_image = tf.random_crop(reshape_image,[24,24,3])
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness = tf.image.random_brightness(flipped_image,max_delta=0.8)
        float_image = tf.image.per_image_standardization(adjusted_brightness)
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_per_epoch_for_eval*0.4)
        print("filling queue with %d images before trainning"%min_queue_examples)
        images_train, labels_train = tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,num_threads=2,
                                                            capacity=min_queue_examples+3*batch_size,min_after_dequeue=min_queue_examples)
        return images_train,tf.reshape(labels_train,[batch_size])
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshape_image,24,24)
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_per_epoch*0.4)
        images_test, labels_test = tf.train.batch([float_image,read_input.label],batch_size=batch_size,num_threads=2,
                                                  capacity=min_queue_examples+3*batch_size)
        return images_test,tf.reshape(labels_test,[batch_size])