#coding=utf-8

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import os
import cv2
from PIL import Image,ImageFile
'''
该文件实现数据的读取
'''
class SimpleDataSet(data.Dataset):
    def __init__(self,rootpath,transforms=None):
        '''
        读取数据集
        rootpath:图像的根目录
        '''
        self.files=os.listdir(rootpath)
        self.classes={"cat":0,"dog":1}
        
        self.all_inputs=[]
        self.all_targets=[]
        for file in self.files:
            abspath=os.path.join(rootpath,file)
            label=int(self.classes[file.split('.')[0]])
            self.all_inputs.append(abspath)
            self.all_targets.append(label)
        self.all_inputs=np.array(self.all_inputs).reshape(len(self.all_inputs),1)
        self.all_targets=np.array(self.all_targets).reshape(self.all_inputs.shape[0],1)
        self.total_inputs=np.concatenate((self.all_inputs,self.all_targets),axis=1)
        
       # print("self.all_inputs:",self.all_inputs.shape)
        self.transforms=transforms#数据增强

    def __getitem__(self,index):
        '''
        返回训练数据
        '''
        #读取数据
       # print(self.total_inputs[index])
        img=cv2.imread(self.total_inputs[index][0])
        label=int(self.total_inputs[index][1])
        label=torch.from_numpy(np.array(label))
        img=np.ascontiguousarray(img[:, :, ::-1])#转为rgb
        if self.transforms is not None:
            img=Image.fromarray(img)
            img=self.transforms(img)
        
        return img,label
    def __len__(self):
        '''
        返回样本总数
        '''
        return len(self.total_inputs)



      