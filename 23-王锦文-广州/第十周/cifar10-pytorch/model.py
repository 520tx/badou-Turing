#!usr/bin/env python  
# -*- coding=utf-8 _*-
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
def conv(in_planes, out_planes, kernel_size=3,padding=1,stride=1, groups=1,bias=True, dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     groups=groups,
                     padding=padding,
                     bias=bias,
                     dilation=dilation)
class CifarSimpleNet(nn.Module):
    def __init__(self, cfg=[64,128],num_classes=10):
        super(CifarSimpleNet, self).__init__()
        fc_scale=8*8
        self.layer1 = nn.Sequential(
                        conv(3,cfg[0],kernel_size=5,padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.layer2 = nn.Sequential(
                        conv(cfg[0],cfg[1],kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.fc1=nn.Linear(cfg[1]*fc_scale,192)
        self.fc2=nn.Linear(192,256)
        self.fc3=nn.Linear(256,num_classes)

        self.softmax=nn.Softmax(dim=1)
        #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x,btrain=True):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        #print("out.shape:",out.shape)#torch.Size([16, 8192])
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if btrain==False:
            out=self.softmax(out)
        return out
def build_model(num_classes=10):
    '''
    构建模型
    '''
    model=CifarSimpleNet(num_classes=num_classes)
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
