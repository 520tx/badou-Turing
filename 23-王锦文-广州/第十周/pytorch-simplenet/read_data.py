#coding=utf-8

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
'''
该文件实现数据的读取
'''
class SimpleDataSet(data.Dataset):
    def __init__(self,datapath,transforms=None):
        '''
        读取数据集
        '''
        with open(datapath,'r') as f:
            self.training_data_list = f.readlines()
        all_inputs=[]
        all_targets=[]
        for line in self.training_data_list:
            values=line.split(",")
            inputs = (np.asfarray(values[1:]))/255.0 * 0.99 + 0.01
            targets=int(values[0])#标签
            all_inputs.append(inputs)
            all_targets.append(targets)
        all_inputs=np.array(all_inputs)
        all_targets=np.array(all_targets)
        all_targets=all_targets.reshape(all_inputs.shape[0],1)
        self.total_inputs=np.concatenate((all_inputs,all_targets),axis=1)
        self.transforms=transforms#数据增强
    def __getitem__(self,index):
        '''
        返回训练数据
        '''
        data=self.total_inputs[index,:-1].astype(np.float32)
        label=self.total_inputs[index,-1]

        data=torch.from_numpy(data)

        return data,label
    def __len__(self):
        '''
        返回样本总数
        '''
        return len(self.total_inputs)



      