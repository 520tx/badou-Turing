#环境中使用python3.7，Keras 2.1.5，
#注意concatenate时的顺序有严格要求否则代码无法执行
import numpy
from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D

from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

def con2d_bn(x,filters,num_row,num_col,stride=(1,1),padding='same',name=None):
    if name is not None:
        bn_name=name+'_bn'
        conv_name=name+'_conv'
    else:
        bn_name=None
        conv_name=None

    x=Conv2D(filters,(num_row,num_col),strides=stride,padding=padding,use_bias=False,name=conv_name)(x)
    x=BatchNormalization(scale=False,name=bn_name)(x)
    x=Activation('relu',name=name)(x)

    return x


def InceptionV3(Input_shape=[299,299,3],classes=1000):
    img=Input(shape=Input_shape)  #输入必须是input_layers或者是input产生的tensor

    x=con2d_bn(img,32,3,3,stride=(2,2),padding='valid')
    x=con2d_bn(x,32,3,3,padding='valid')
    x=con2d_bn(x,64,3,3)
    x=MaxPooling2D((3,3),strides=(2,2))(x)

    x=con2d_bn(x,80,1,1,padding='valid')
    x=con2d_bn(x,192,3,3,padding='valid')
    x=MaxPooling2D((3,3),strides=(2,2))(x)

  #  3xInception  part_one
    branch1x1=con2d_bn(x,64,1,1)

    branch_conv5x5=con2d_bn(x,48,1,1)
    branch_conv5x5=con2d_bn(branch_conv5x5,64,5,5)

    branch_con3x3=con2d_bn(x,64,1,1)
    branch_con3x3=con2d_bn(branch_con3x3,96,3,3)
    branch_con3x3=con2d_bn(branch_con3x3,96,3,3)

    branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool=con2d_bn(branch_pool,32,1,1)

    x=layers.concatenate([branch1x1,branch_conv5x5,branch_con3x3,branch_pool],axis=3,name='mixed0')

   #part_two
    branch1x1 = con2d_bn(x, 64, 1, 1)

    branch_conv5x5 = con2d_bn(x, 48, 1, 1)
    branch_conv5x5 = con2d_bn(branch_conv5x5, 64, 5, 5)

    branch_con3x3 = con2d_bn(x, 64, 1, 1)
    branch_con3x3 = con2d_bn(branch_con3x3, 96, 3, 3)
    branch_con3x3 = con2d_bn(branch_con3x3, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = con2d_bn(branch_pool, 64, 1, 1)

    x = layers.concatenate([branch1x1, branch_conv5x5,  branch_con3x3,branch_pool],axis=3,name='mixed1')

   #part_three
    branch1x1 = con2d_bn(x, 64, 1, 1)

    branch_conv5x5 = con2d_bn(x, 48, 1, 1)
    branch_conv5x5 = con2d_bn(branch_conv5x5, 64, 5, 5)

    branch_con3x3 = con2d_bn(x, 64, 1, 1)
    branch_con3x3 = con2d_bn(branch_con3x3, 96, 3, 3)
    branch_con3x3 = con2d_bn(branch_con3x3, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = con2d_bn(branch_pool, 64, 1, 1)

    x = layers.concatenate([branch1x1, branch_conv5x5, branch_con3x3,branch_pool],axis=3,name='mixed2')

    # 5xInception  part_1
    block2_conv3x3=con2d_bn(x,384,3,3,stride=(2,2),padding='valid')

    block2_conv3x3dl=con2d_bn(x,64,1,1)
    block2_conv3x3dl=con2d_bn(block2_conv3x3dl,96,3,3)
    block2_conv3x3dl=con2d_bn(block2_conv3x3dl,96,3,3,stride=(2,2),padding='valid')

    block2_pool=MaxPooling2D((3,3),strides=(2,2))(x)
    x=layers.concatenate([block2_conv3x3,block2_conv3x3dl,block2_pool],axis=3,name='mixed3')
    # 5xInception  part_2
    block2_conv1x1=con2d_bn(x,192,1,1)

    block2_con7x7=con2d_bn(x,128,1,1)
    block2_con7x7=con2d_bn(block2_con7x7,128,1,7)
    block2_con7x7=con2d_bn(block2_con7x7,192,7,1)

    block2_con7x7dl = con2d_bn(x,128,1,1)
    block2_con7x7dl = con2d_bn(block2_con7x7dl,128,7,1)
    block2_con7x7dl = con2d_bn(block2_con7x7dl,128,1,7)
    block2_con7x7dl = con2d_bn(block2_con7x7dl, 128, 7, 1)
    block2_con7x7dl = con2d_bn(block2_con7x7dl, 192, 1, 7)

    block2_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool=con2d_bn(block2_pool,192,1,1)

    x=layers.concatenate([block2_conv1x1,block2_con7x7,block2_con7x7dl,branch_pool],axis=3,name='mixed4')
    # 5xInception  part_3/4
    for i in range(2):
        block2_conv1x1=con2d_bn(x,192,1,1)

        block2_con7x7=con2d_bn(x,160,1,1)
        block2_con7x7=con2d_bn(block2_con7x7,160,1,7)
        block2_con7x7=con2d_bn(block2_con7x7,192,7,1)

        block2_con7x7dl=con2d_bn(x,160,1,1)
        block2_con7x7dl=con2d_bn(block2_con7x7dl,160,7,1)
        block2_con7x7dl=con2d_bn(block2_con7x7dl,160,1,7)
        block2_con7x7dl=con2d_bn(block2_con7x7dl,160,7,1)
        block2_con7x7dl=con2d_bn(block2_con7x7dl,192,1,7)

        block2_pool=MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
        block2_pool=con2d_bn(block2_pool,192,1,1)

        x=layers.concatenate([block2_conv1x1,block2_con7x7,block2_con7x7dl,block2_pool],axis=3,name='mixed'+str(i+5))

        # 5xInception  part_5
    block2_conv1x1=con2d_bn(x,192,1,1)

    block2_con7x7=con2d_bn(x,192,1,1)
    block2_con7x7=con2d_bn(block2_con7x7,192,1,7)
    block2_con7x7=con2d_bn(block2_con7x7,192,7,1)


    block2_con7x7dl=con2d_bn(x,192,1,1)
    block2_con7x7dl=con2d_bn(block2_con7x7dl,192,7,1)
    block2_con7x7dl=con2d_bn(block2_con7x7dl,192,1,7)
    block2_con7x7dl=con2d_bn(block2_con7x7dl,192,7,1)
    block2_con7x7dl=con2d_bn(block2_con7x7dl,192,1,7)

    block2_pool=MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
    block2_pool=con2d_bn(block2_pool,192,1,1)

    x=layers.concatenate([block2_conv1x1,block2_con7x7,block2_con7x7dl,block2_pool],axis=3,name='mixed7')

   # 2xblcok_3 part_1
    block3_conv3x3=con2d_bn(x,192,1,1)
    block3_conv3x3=con2d_bn(block3_conv3x3,320,3,3,stride=(2,2),padding='valid')

    block3_conv7x7=con2d_bn(x,192,1,1)
    block3_conv7x7=con2d_bn(block3_conv7x7,192,1,7)
    block3_conv7x7=con2d_bn(block3_conv7x7,192,7,1)
    block3_conv7x7=con2d_bn(block3_conv7x7,192,3,3,stride=(2,2),padding='valid')

    block3_pool=MaxPooling2D((3,3),strides=(2,2))(x)
    x=layers.concatenate([block3_conv3x3,block3_conv7x7,block3_pool],axis=3,name='mixed8')

    for i in range(2):
        block3_conv1x1=con2d_bn(x,320,1,1)

        block3_conv3x3 = con2d_bn(x,384,1,1)
        block3_conv3x3_1 = con2d_bn(block3_conv3x3,384,1,3)
        block3_conv3x3_2 = con2d_bn(block3_conv3x3,384,3,1)
        block3_conv3x3 = layers.concatenate([block3_conv3x3_1,block3_conv3x3_2],axis=3,name='mixed9_'+str(i))

        block3_conv3x3dl=con2d_bn(x,448,1,1)
        block3_conv3x3dl=con2d_bn(block3_conv3x3dl,384,3,3)
        block3_conv3x3dl_1=con2d_bn(block3_conv3x3dl,384,1,3)
        block3_conv3x3dl_2=con2d_bn(block3_conv3x3dl,384,3,1)
        block3_conv3x3dl=layers.concatenate([block3_conv3x3dl_1,block3_conv3x3dl_2],axis=3)

        block3_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        block3_pool=con2d_bn(block3_pool,192,1,1)

        x=layers.concatenate([block3_conv1x1,block3_conv3x3,block3_conv3x3dl,block3_pool],axis=3,name='mixed'+str(i+9))

    x=GlobalAveragePooling2D(name='avg_pool')(x)
    x=Dense(classes,activation='softmax',name='predictions')(x)

    inputs=img
    model=Model(inputs,x,name='inception_v3')
    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
if __name__ =='__main__':
    model=InceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    image_path='elephant.jpg'
    img=image.load_img(image_path,target_size=(299,299))

    x=image.img_to_array(img)
    x=numpy.expand_dims(x,axis=0)

    x=preprocess_input(x)
    preds=model.predict(x)
    print(decode_predictions(preds))