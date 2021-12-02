#!usr/bin/env python  
# -*- coding=utf-8 _*-
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
'''
该py文件实现详细的mobilenetv1
'''
class BasicConv(nn.Module):
    '''
    普通conv+bn+relu
    '''
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5,) if bn else None
        self.relu = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicdwConv(nn.Module):
    '''
    深度可分离卷积实现
    '''
    def __init__(self, in_planes, out_planes,kernel_size, stride=1, padding=0,dilation=1,bias=False):
        super(BasicdwConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, )
        self.relu = nn.ReLU6(inplace=True)
        self.conv1=nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes,eps=1e-5, )
        self.relu1= nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x
class mobilenetv1(nn.Module):
    def __init__(self,num_classes=2):
        super(mobilenetv1,self).__init__()
        self.bacbone=nn.Sequential(
            BasicConv(3,32,kernel_size=3,stride=2,padding=1),#第一层卷积,stride=2 1/2
            BasicdwConv(32,64,kernel_size=3,stride=1,padding=1),#
            BasicdwConv(64,128,kernel_size=3,stride=2,padding=1), #1/4
            BasicdwConv(128,128,kernel_size=3,stride=1,padding=1),
            BasicdwConv(128,256,kernel_size=3,stride=2,padding=1), #1/8
            BasicdwConv(256,256,kernel_size=3,stride=1,padding=1),
            BasicdwConv(256,512,kernel_size=3,stride=2,padding=1),#1/16
            BasicdwConv(512,512,kernel_size=3,stride=1,padding=1),
            BasicdwConv(512,512,kernel_size=3,stride=1,padding=1),
            BasicdwConv(512,512,kernel_size=3,stride=1,padding=1),
            BasicdwConv(512,512,kernel_size=3,stride=1,padding=1),
            BasicdwConv(512,512,kernel_size=3,stride=1,padding=1),
            BasicdwConv(512,1024,kernel_size=3,stride=2,padding=1),#1/32
            BasicdwConv(1024,1024,kernel_size=3,stride=1,padding=1),
        )
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(1024,num_classes)
        #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        out=self.bacbone(x)
        out=self.avg_pool(out).view(out.size(0),-1)
        out=self.fc(out)
        return out
def build_mobilenetv1(num_classes=2):
    model=mobilenetv1(num_classes=num_classes)
    return model

        
def load_model(model, model_path, optimizer=None,resume=False, 
               lr=None, lr_step=None,device='cpu',lr_sched=None):

    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict'] 
    state_dict = {}

    # 并行训练保存的模型有.module
    for k in state_dict_:
        if k.startswith('module'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_dict = model.state_dict()#当前模型的字典
    model_state_dict={}
    for k,v in state_dict.items():
        if k in model_dict.keys():
            model_state_dict[k]=v
            #print("k:",k)
    model_dict.update(model_state_dict)

    model.load_state_dict(model_dict)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device, non_blocking=True)
            start_epoch = checkpoint['epoch']
            #学习率
           
            start_lr = lr_sched(start_epoch)
            # for step in lr_step:
            #     if start_epoch >= step:
            #         start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
    if optimizer is not None:
        print("START EPOCH===:",start_epoch)
        return model, optimizer, start_epoch
    else:
        return model

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
