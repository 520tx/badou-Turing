#!usr/bin/env python  
# -*- coding=utf-8 _*-
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
def conv3x3(in_planes, out_planes, kernel_size=3,padding=1,stride=1, groups=1,bias=False, dilation=1):
    '''
    3x3卷积
    '''
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     groups=groups,
                     padding=padding,
                     bias=bias,
                     dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
class BasicBlock(nn.Module):
    '''
    resnet的基础block
    '''
    expansion=4
    def __init__(self,in_channels,outchannels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1x1_1=conv1x1(in_channels,outchannels)
        self.bn1=nn.BatchNorm2d(outchannels,eps=1e-5)
        self.relu1=nn.ReLU(inplace=True)
        self.conv3x3_1=conv3x3(outchannels,outchannels,stride=stride)
        self.bn2=nn.BatchNorm2d(outchannels,eps=1e-5)
        self.relu2=nn.ReLU(inplace=True)
        self.conv1x1_2=conv1x1(outchannels,outchannels*self.expansion)
        self.bn3=nn.BatchNorm2d(outchannels*self.expansion,eps=1e-5)
        self.stride=stride
        self.downsample=downsample
        self.relu3=nn.ReLU(inplace=True)

    def forward(self,x):
        identity=x
        out=self.conv1x1_1(x)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.conv3x3_1(out)
        out=self.bn2(out)
        out=self.relu2(out)
        out=self.conv1x1_2(out)
        out=self.bn3(out)
        if self.downsample is not None:
            identity=self.downsample(x)
        out+=identity
        out=self.relu3(out)
        return out
class ResNet(nn.Module):
   
    def __init__(self,block,layers,dropout=0,num_classes=2):
        super(ResNet,self).__init__()
        self.inplanes=64
        self.conv1=conv3x3(3,self.inplanes,kernel_size=7,padding=3,stride=2)
        
        self.bn1=nn.BatchNorm2d(self.inplanes,eps=1e-5)
        self.relu1=nn.ReLU(inplace=True)
        self.Maxpool=nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1=self.make_layer(block,64,layers[0],stride=1)
        self.layer2=self.make_layer(block,128,layers[1],stride=2)
        self.layer3=self.make_layer(block,256,layers[2],stride=2)
        self.layer4=self.make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc=nn.Linear(2048,num_classes)
        self.softmax=nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x,is_train=True):
        '''
        resnet 前向传播
        '''
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.Maxpool(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)#[batch_size,len(feature)]
        out=self.fc(out)
        if is_train==False:
            out=self.softmax(out)
        
        return out


    def make_layer(self,block,planes,blocks,stride=1):
        '''
        resnet的每个层级包含的若干个block
        '''
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride=stride),
                nn.BatchNorm2d(planes*block.expansion,eps=1e-5)
            )
        layers=[]
        layers.append(block(self.inplanes,planes,stride=stride,downsample=downsample))#先进行一次下采样(stride=2的情况才下采样，否则不变)
        self.inplanes=planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)


def build_Resnet50(num_classes=2):
    model=ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes)
    return model

        
def load_model(model, model_path, optimizer=None,resume=False, 
               lr=None, lr_step=None,device='cpu'):

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
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
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
