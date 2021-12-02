#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：cifar10_conv.py
@Author  ：luigi
@Date    ：2021/9/9 12:21 下午 
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def get_data(image_path, download=False):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(image_path, train=True, download=download, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(image_path, train=False, download=download)
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader,testloader

def get_model():

    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 512, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(36*512,10)
    )
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-1)

    return model, loss_fn, optimizer

def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    prediction = model(x)

    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y

    return is_correct.numpy().tolist()

def main():
    epoch_num = 5
    image_path = '/Users/liujie/工作相关/data/CIFAR10'
    train_dl,test_dl = get_data(image_path)
    model, loss_fn, opt = get_model()

    losses, accuracies = [], []

    for i in range(epoch_num):
        print("epoch {}: ".format(i))

        epoch_losses, epoch_accuracies = [], []

        for ix, batch in enumerate(iter(train_dl)):
            x, y = batch

            batch_loss = train_batch(x, y, model, loss_fn, opt)
            epoch_losses.append(batch_loss)
        epoch_loss = np.mean(epoch_losses)

        for ix, batch in enumerate(iter(train_dl)):
            x, y = batch

            is_correct = accuracy(x, y, model)
            epoch_accuracies.append(is_correct)
        epoch_accuracy = np.mean(epoch_accuracies)
        print("the epoch_accuracy is {}: ".format(epoch_accuracy))

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

    epochs = np.arange(epoch_num) + 1
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('Loss value over incressing epoch')
    plt.plot(epochs, losses, label='training loss')
    plt.subplot(122)
    plt.title('accuracy value over incressing epoch')
    plt.plot(epochs, accuracies, label='training accuracy')
    plt.show()


if __name__ == '__main__':
    main()












