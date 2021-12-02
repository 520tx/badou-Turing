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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#cifar10数据集标签类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def test(args):
    '''
    对cifar10测试集进行测试
    '''
    # 获取数据增强方式
    train_tsfm, valid_tsfm = get_transform(32, 32)
    # 测试集
    test_dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=False,
                                                download=True, transform=valid_tsfm)
    # 创建dataloader
    val_dataloders = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,  #
        drop_last=False
    )
    # 创建模型
    model = build_model(args.numclasses)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model, args.load_model, device=device)
    model.eval()
    model.to(device)

    # 分别计算每类的准确率
    correct_ = {classname: 0 for classname in classes}
    total_= {classname: 0 for classname in classes}
    for i, (inputs, labels) in enumerate(val_dataloders):
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        # forward
        outputs = model(inputs, btrain=False).detach().cpu()
        _, preds = torch.max(outputs, 1)
        # print("pred:",type(preds),preds.shape)#torch.Tensor,torch.size([16])
        labels = labels.cpu()
        for i in range(args.numclasses):
            correct_[classes[i]] += torch.sum((preds == i) * (preds == labels), axis=0)  # 每类预测正确的个数
            # 每类的总数累计
            total_[classes[i]] += torch.sum((labels == i), axis=0)
    # 计算每类的准确率
    for classname, correct_cnt in correct_.items():
        accuracy = 100 * float(correct_cnt) / total_[classname]
        print("class {} acc is: {:.4f} %".format(classname,accuracy))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 test')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch_size')
    parser.add_argument('--numclasses', default=10, type=int,
                        help='how many classes')
    parser.add_argument('--datapath', default='./cifar-10-batches-bin/', type=str,
                        help='where is the test dataset')
    parser.add_argument('--load_model', default='./model_last.pth',
                        help='path to pretrained model')
    args=parser.parse_args()
    test(args)
