#!usr/bin/env python  
# -*- coding=utf-8 _*-
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
import os
import torchvision
#from model import build_InceptionV3,load_model,save_model
from mobilenetv1 import build_mobilenetv1,load_model,save_model
import argparse

from read_dataset import SimpleDataSet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

'''
该文件使用pytorch1.2简单实现inceptionv3的的2分类(cat and dog)训练
'''

def train(args):
    #获取数据增强方式
    train_tsfm, valid_tsfm = get_transform(224,224)
    #训练集
    train_dataset=SimpleDataSet(args.datapath,transforms=train_tsfm)

    #创建dataloader
    train_dataloders=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,#是否将数据保存在pin_memory ,这样转GPU会快一点,这里数据较少，用False
        drop_last=True
        )
    #创建模型
    model=build_mobilenetv1(num_classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
   # device=torch.device("cpu")

    #定义loss
    criterion=nn.CrossEntropyLoss()
    #定义优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00005)
    
    #定义学习率策略
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=0.00005, last_epoch=-1)  #T_max周期迭代数,eta_min最小学习率
    #是否接着训练
    start_epoch=0
    lr = args.lr

    if args.resume:
        model, optimizer, start_epoch = load_model(model, args.load_model, optimizer, args.resume, args.lr, args.lr_step,device,lr_sched=exp_lr_scheduler)
    model.to(device=device)
    for epoch in range(start_epoch,args.epochs):
        mean_loss=0.#
        
        #学习率调整策略
        # for step in args.lr_step:
        #     if epoch >= step:
        #         lr *= 0.1

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for i,(inputs,labels) in enumerate(train_dataloders):
            inputs=inputs.to(device)
            
            labels=labels.long().to(device)
           # print("label:",type(labels))

            #forward
            outputs=model(inputs)

            loss=criterion(outputs,labels)
            loss.backward()
            #梯度更新
            optimizer.step()
            exp_lr_scheduler.step(epoch)
            optimizer.zero_grad()
          #  lr_scheduler.step(epoch)
            mean_loss+=loss.item()
            if i%150==0 and i>0:
                print("epoch={},batchloss={}".format(epoch,mean_loss/150))
                mean_loss = 0.0
        
        save_model(os.path.join(args.savepath, 'model_last.pth'),
                   epoch, model, optimizer)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch resnet50 train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--cuda', default=True,type=bool,
                             help='use gpu')
    parser.add_argument('--epochs', default=20, type=int,
                             help='how many epochs to train')
    parser.add_argument('--batch_size', default=32, type=int,
                             help='batch_size')
    parser.add_argument('--numclasses', default=2, type=int,
                             help='how many classes')
    parser.add_argument('--datapath', default='./train/', type=str,
                             help='where is the train dataset for cat and dog ')
    parser.add_argument('--lr_step', type=str, default='7,12,16',
                        help='drop learning rate by 10.')
    parser.add_argument('--savepath', type=str, default='./',
                        help='savemodel path')
    parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    
    args = parser.parse_args()
    args.lr_step = [int(i) for i in args.lr_step.split(',')]
    train(args)



            



    
    


