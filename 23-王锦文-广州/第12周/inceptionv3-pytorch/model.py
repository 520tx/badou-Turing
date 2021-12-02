#!usr/bin/env python  
# -*- coding=utf-8 _*-
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
'''
该py文件实现详细的inceptionv3
'''
class BasicConv(nn.Module):
    '''
    基础模块conv+bn+relu
    '''
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5,) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class inceptionV3blob1(nn.Module):
    '''inceptionv3的第一种blob结构'''
    """
    在inceptionv3中这样的blob共有3组这样的结构
    """
    def __init__(self,in_planes,bfirst=False):
        super(inceptionV3blob1,self).__init__()
        if bfirst==True:#这里第一个blob接pooling后在进行32通道的卷积,其余为64通道
            self.b4_chn=32
        else:
            self.b4_chn=64
        self.branch1=BasicConv(in_planes=in_planes,out_planes=64,kernel_size=1)
        self.branch2=nn.Sequential(
            BasicConv(in_planes=in_planes,out_planes=48,kernel_size=1),
            BasicConv(in_planes=48,out_planes=64,kernel_size=5,padding=2)
        )
        self.branch3=nn.Sequential(
            BasicConv(in_planes=in_planes,out_planes=64,kernel_size=1),
            BasicConv(in_planes=64,out_planes=96,kernel_size=3,padding=1),
            BasicConv(in_planes=96,out_planes=96,kernel_size=3,padding=1)
        )
        self.branch4=nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_planes=in_planes,out_planes=self.b4_chn,kernel_size=1,padding=0)
        )
    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out4=self.branch4(x)
        out=torch.cat([out1,out2,out3,out4],dim=1)
        return out

class inceptionV3blob2(nn.Module):
    '''
    v3的第二种blob，有下采样
    '''
    def __init__(self,in_planes):
        super(inceptionV3blob2,self).__init__()
        self.branch1=nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2=BasicConv(in_planes=in_planes,out_planes=384,kernel_size=3,stride=2,padding=0)
        self.branch3=nn.Sequential(
            BasicConv(in_planes=288,out_planes=64,kernel_size=1),
            BasicConv(in_planes=64,out_planes=96,kernel_size=3,padding=1),
            BasicConv(in_planes=96,out_planes=96,kernel_size=3,stride=2,padding=0)
        )
    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out=torch.cat([out1,out2,out3],dim=1)
        return out

class inceptionV3blob3(nn.Module):
    '''
    v3的第三种blob
    '''
    def __init__(self,in_planes,out_planes,tmpchannel=128):
        super(inceptionV3blob3,self).__init__()
        self.branch1=BasicConv(in_planes=in_planes,out_planes=out_planes,kernel_size=1,padding=0)
        self.branch2=nn.Sequential(
            BasicConv(in_planes=in_planes,out_planes=tmpchannel,kernel_size=1),
            BasicConv(in_planes=tmpchannel,out_planes=tmpchannel,kernel_size=(1,7),padding=(0,3)),
            BasicConv(in_planes=tmpchannel,out_planes=out_planes,kernel_size=(7,1),padding=(3,0))
        )
        self.branch3=nn.Sequential(
            BasicConv(in_planes=in_planes,out_planes=tmpchannel,kernel_size=1),
            BasicConv(in_planes=tmpchannel,out_planes=tmpchannel,kernel_size=(7,1),padding=(3,0)),
            BasicConv(in_planes=tmpchannel,out_planes=tmpchannel,kernel_size=(1,7),padding=(0,3)),
            BasicConv(in_planes=tmpchannel,out_planes=tmpchannel,kernel_size=(7,1),padding=(3,0)),
            BasicConv(in_planes=tmpchannel,out_planes=out_planes,kernel_size=(1,7),padding=(0,3))
        )
       
        self.branch4=nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_planes=in_planes,out_planes=out_planes,kernel_size=1,padding=0)
        )
    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out4=self.branch4(x)
        out=torch.cat([out1,out2,out3,out4],dim=1)
        return out

class inceptionV3blob4(nn.Module):
    '''
    v3的第4种blob，有下采样
    '''
    def __init__(self,in_planes):
        super(inceptionV3blob4,self).__init__()
        self.branch1=nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2=nn.Sequential(
            BasicConv(in_planes=in_planes,out_planes=192,kernel_size=1),
            BasicConv(in_planes=192,out_planes=320,kernel_size=3,stride=2,padding=0),
         
        )
        self.branch3=nn.Sequential(
            BasicConv(in_planes=in_planes,out_planes=192,kernel_size=1),
            BasicConv(in_planes=192,out_planes=192,kernel_size=(1,7),padding=(0,3)),
            BasicConv(in_planes=192,out_planes=192,kernel_size=(7,1),padding=(3,0)),
            BasicConv(in_planes=192,out_planes=192,kernel_size=3,stride=2,padding=0),
        )
    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out=torch.cat([out1,out2,out3],dim=1)
        return out

class inceptionV3blob5(nn.Module):
    '''
    v3的第5种blob
    '''
    def __init__(self,in_planes,tmpchannel=[320,384,448,192]):
        super(inceptionV3blob5,self).__init__()
        self.branch1=BasicConv(in_planes=in_planes,out_planes=tmpchannel[0],kernel_size=1,padding=0)
        self.branch2_1x1=BasicConv(in_planes=in_planes,out_planes=tmpchannel[1],kernel_size=1)
        self.branch2_1x3= BasicConv(in_planes=tmpchannel[1],out_planes=tmpchannel[1],kernel_size=(1,3),padding=(0,1))
        self.branch2_3x1= BasicConv(in_planes=tmpchannel[1],out_planes=tmpchannel[1],kernel_size=(3,1),padding=(1,0))
        
        self.branch3_1x1=BasicConv(in_planes=in_planes,out_planes=tmpchannel[2],kernel_size=1)
        self.branch3_3x3=BasicConv(in_planes=tmpchannel[2],out_planes=tmpchannel[1],kernel_size=3,padding=1)
        self.branch3_1x3= BasicConv(in_planes=tmpchannel[1],out_planes=tmpchannel[1],kernel_size=(1,3),padding=(0,1))
        self.branch3_3x1= BasicConv(in_planes=tmpchannel[1],out_planes=tmpchannel[1],kernel_size=(3,1),padding=(1,0))
        
   
        self.branch4=nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_planes=in_planes,out_planes=tmpchannel[3],kernel_size=1,padding=0)
        )
    def forward(self,x):
        out1=self.branch1(x)
        out2_1=self.branch2_1x1(x)
        out2_2=self.branch2_1x3(out2_1)
        out2_3=self.branch2_3x1(out2_1)
        out2=torch.cat([out2_2,out2_3],dim=1)
        out3_1=self.branch3_1x1(x)
        out3_2=self.branch3_3x3(out3_1)
        out3_3=self.branch3_1x3(out3_2)
        out3_4=self.branch3_3x1(out3_2)
        out3=torch.cat([out3_3,out3_4],dim=1)
        out4=self.branch4(x)
        out=torch.cat([out1,out2,out3,out4],dim=1)
        return out

class inceptionV3(nn.Module):
    def __init__(self,num_classes=2):
        super(inceptionV3,self).__init__()
        self.stage_0=nn.Sequential(
            BasicConv(in_planes=3,out_planes=32,kernel_size=3,stride=2,padding=0),
            BasicConv(in_planes=32,out_planes=32,kernel_size=3,stride=1),
            BasicConv(in_planes=32,out_planes=64,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv(in_planes=64,out_planes=80,kernel_size=1,stride=1),
            BasicConv(in_planes=80,out_planes=192,kernel_size=3,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.stage_1=nn.Sequential(
            inceptionV3blob1(in_planes=192,bfirst=True),#第一个blob的第4个分支输出通道为32
            inceptionV3blob1(in_planes=256,bfirst=False),
            inceptionV3blob1(in_planes=288,bfirst=False)
        )
        self.down_sample1=inceptionV3blob2(in_planes=288)
        self.stage_2=nn.Sequential(
            inceptionV3blob3(in_planes=768,out_planes=192,tmpchannel=128),
            inceptionV3blob3(in_planes=768,out_planes=192,tmpchannel=160),
            inceptionV3blob3(in_planes=768,out_planes=192,tmpchannel=160),
            inceptionV3blob3(in_planes=768,out_planes=192,tmpchannel=192)
        )
        self.down_sample2=inceptionV3blob4(in_planes=768)
        self.stage_3=nn.Sequential(
            inceptionV3blob5(in_planes=1280),
            inceptionV3blob5(in_planes=2048)
        )
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        #使用1*1卷积代替全连接
        self.linear=nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, padding=0,bias=True)

        self.softmax=nn.Softmax(dim=1)
        #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x,is_train=True):
        out=self.stage_0(x)#在处理完这个stage后，特征图尺寸已经减少到了1/8
        out=self.stage_1(out)
        out=self.down_sample1(out)#1/16
        out=self.stage_2(out)
        out=self.down_sample2(out)#1/32
        out=self.stage_3(out)
        out=self.avg_pool(out)
        out=self.linear(out).view(x.size(0), -1)
        if is_train==False:
            out=self.softmax(out)
        return out

def build_InceptionV3(num_classes=2):
    model=inceptionV3(num_classes=num_classes)
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
