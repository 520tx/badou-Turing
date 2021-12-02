#!usr/bin/env python
# -*- coding=utf-8 _*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import os
import torchvision
from model import build_Resnet50,load_model,save_model
import argparse
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

classes ={0:'cat',1:'dog'}

def test(args):
    '''
    对单张图片测试
    '''
      
    # 创建模型
    model = build_Resnet50(args.numclasses)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model, args.load_model, device=device)
    model.eval()
    model.to(device)

    mean=np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(1,1,3)
    std=np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(1,1,3)
    img=cv2.imread(args.imgpath)
    showimg=img.copy()
    
    img=cv2.resize(img,(224,224))
    inp_img=np.ascontiguousarray(img[:, :, ::-1])
    inp_img = (inp_img / 255.).astype(np.float32)
    inp_img-=mean
    inp_img/=std
    inp_img = inp_img.transpose(2, 0, 1).reshape(1, 3, 224, 224)
    inp_img=torch.from_numpy(inp_img).to(device)
    preds=model(inp_img,is_train=False).detach().cpu()
    print("preds:",preds.shape)
    predlabel=np.argmax(preds,1).item()
    print(predlabel)
    cv2.putText(showimg,"{}".format(classes[predlabel]),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
    cv2.imshow("result",showimg)
    cv2.waitKey(0)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch resnet50 test')
    parser.add_argument('--numclasses', default=2, type=int,
                        help='how many classes')
    parser.add_argument('--imgpath', default='./test/dog2.jpg', type=str,
                        help='where is the test image')
    parser.add_argument('--load_model', default='./model_last.pth',
                        help='path to pretrained model')
    args=parser.parse_args()
    test(args)
