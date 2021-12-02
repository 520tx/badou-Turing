import numpy

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as k
def relu_6(x):
    return k.relu(x,max_value=6)
def conv_block(inputs,filters,kernel=(3,3),strides=(1,1)):
    x=Conv2D(filters,kernel,strides=strides,padding='same',use_bias=False,name='conv1')(inputs)
    x=BatchNormalization(name='conv1_bn')(x)
    return Activation(relu_6,name='conv1_relu')(x)

def depthwise_con_block(inputs,pointwise_conv_filters,depth_multiplier=1,strides=(1,1),block_id=1):

    x=DepthwiseConv2D((3,3),padding='same',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_%d' %block_id)(inputs)
    x=BatchNormalization(name='conv_dw_%d_bn'%block_id)(x)
    x=Activation(relu_6,name='conv_dw_%d_relu'%block_id)(x)
    x=Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,strides=(1,1),name='conv_pw_%d'%block_id)(x)
    x=BatchNormalization(name='conv_pw_%d_bn'%block_id)(x)
    return Activation(relu_6,name='conv_pw_%d_relu'%block_id)(x)

def MobileNet(input_shape=[224,224,3],depth_multiplier=1,dropout=1e-3,classes=1000):
    img_input=Input(shape=input_shape)

    x=conv_block(img_input,32,strides=(2,2))
    x=depthwise_con_block(x,pointwise_conv_filters=64,block_id=1)
    x=depthwise_con_block(x,pointwise_conv_filters=128,strides=(2,2),block_id=2)
    x=depthwise_con_block(x,pointwise_conv_filters=128,block_id=3)
    x=depthwise_con_block(x,pointwise_conv_filters=256,strides=2,block_id=4)
    x=depthwise_con_block(x,pointwise_conv_filters=256,block_id=5)
    x=depthwise_con_block(x,pointwise_conv_filters=512,strides=(2,2),block_id=6)

    for i in range(5):
        x=depthwise_con_block(x,pointwise_conv_filters=512,block_id=(i+7))
    x=depthwise_con_block(x,pointwise_conv_filters=1024,depth_multiplier=1,strides=(2,2),block_id=12)
    x=depthwise_con_block(x,pointwise_conv_filters=1024,depth_multiplier=1,strides=(2,2),block_id=13)
    x=GlobalAveragePooling2D()(x)  #全局平均池化之后输出为1x1x1024
    x=Reshape((1,1,1024),name='reshape_1')(x)
    x=Conv2D(classes,(1,1),padding='same',name='FC_conv')(x)
    x=Activation('softmax',name='act_softmax')(x)
    x=Reshape((classes,),name='reshape_2')(x)

    inputs=img_input
    model=Model(inputs,x,name='MobileNet_applcation')
    model_path='mobilenet_1_0_224_tf.h5'
    model.load_weights(model_path)
    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__=='__main__':
    model=MobileNet()

    img_path='elephant.jpg'
    img=image.load_img(img_path,target_size=(224,224,3))
    x=image.img_to_array(img)
    x=numpy.expand_dims(x,axis=0)
    x=preprocess_input(x)
    preds=model.predict(x)
    print(decode_predictions(preds,1))