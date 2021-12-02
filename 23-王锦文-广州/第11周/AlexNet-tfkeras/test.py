
import numpy as np
import tensorflow as tf
import cv2
from model import *
import os
import argparse

#K.set_image_dim_ordering('tf')
classes={0:'cat',1:"dog"}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='keras demo test')
    parser.add_argument('--datapath', default='./test/', type=str,
                             help='where is the test dataset')
    parser.add_argument('--load_model', default='model_last.h5',type=str,
                             help='path to pretrained model')
    inputsize=(224,224)
    args = parser.parse_args()
    model = build_AlexNet()
    model.load_weights(args.load_model)
    files=os.listdir(args.datapath)
    for file in files:
        image=cv2.imread(os.path.join(args.datapath,file))
        showimg=image.copy()
        if (image.shape[0]!=224 or image.shape[1]!=224):
            image=cv2.resize(image,inputsize)
        image=np.ascontiguousarray(image[:, :, ::-1])
        image=image.astype(np.float32)/255.
        image = np.expand_dims(image,axis = 0)
        pred=np.argmax(model.predict(image))
        cv2.putText(showimg,"{}".format(classes[pred]),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imshow("result",showimg)
        cv2.waitKey(0)