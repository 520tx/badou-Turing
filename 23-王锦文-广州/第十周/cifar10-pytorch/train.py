#!usr/bin/env python  
# -*- coding=utf-8 _*-
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from utils import *
import os
import torchvision
from model import build_model,load_model,save_model
import argparse

from read_data import SimpleDataSet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

'''
该文件使用pytorch1.2简单实现CIFAR-10的训练和测试
'''

def train(args):
    #获取数据增强方式
    train_tsfm, valid_tsfm = get_transform(32,32)
    #训练集 
    train_dataset=torchvision.datasets.CIFAR10(root=args.datapath, train=True,
                                        download=True, transform=train_tsfm)
    #测试集
    test_dataset=torchvision.datasets.CIFAR10(root=args.datapath, train=False,
                                        download=True, transform=valid_tsfm)
                
    #创建dataloader
    train_dataloders=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,#是否将数据保存在pin_memory ,这样转GPU会快一点,这里数据较少，用False
        drop_last=True
        )
    val_dataloders=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,#
        drop_last=False
        )
    #创建模型
    model=build_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
   # device=torch.device("cpu")

    #定义loss
    criterion=nn.CrossEntropyLoss()
    #定义优化器
    optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=0.00004)
    
    #是否接着训练
    start_epoch=0
    lr = args.lr
    if args.resume:
        model, optimizer, start_epoch = load_model(model, args.load_model, optimizer, args.resume, args.lr, args.lr_step,device)
    model.to(device=device)
    for epoch in range(start_epoch,args.epochs):
        mean_loss=0.#
        
        #学习率调整策略
        for step in args.lr_step:
            if epoch >= step:
                lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for i,(inputs,labels) in enumerate(train_dataloders):
            inputs=inputs.to(device)
            labels=labels.to(device)
            #forward
            outputs=model(inputs)

            loss=criterion(outputs,labels)
            loss.backward()
            #梯度更新
            optimizer.step()
            optimizer.zero_grad()
          #  lr_scheduler.step(epoch)
            mean_loss+=loss.item()
            if i%100==0 and i>0:
                print("epoch={},batchloss={}".format(epoch,mean_loss/100))
                mean_loss = 0.0
        save_model(os.path.join(args.savepath, 'model_last.pth'),
                   epoch, model, optimizer)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--cuda', default=True,type=bool,
                             help='use gpu')
    parser.add_argument('--epochs', default=18, type=int,
                             help='how many epochs to train')
    parser.add_argument('--batch_size', default=32, type=int,
                             help='batch_size')
    parser.add_argument('--numclasses', default=10, type=int,
                             help='how many classes')
    parser.add_argument('--datapath', default='./cifar-10-batches-bin/', type=str,
                             help='where is the train dataset')
    parser.add_argument('--lr_step', type=str, default='9,14',
                        help='drop learning rate by 10.')
    parser.add_argument('--savepath', type=str, default='./',
                        help='savemodel path')
    parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    
    args = parser.parse_args()
    args.lr_step = [int(i) for i in args.lr_step.split(',')]
    train(args)



            



    
    


