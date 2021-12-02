#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：CatDogClassifier.py
@Author  ：luigi
@Date    ：2021/9/1 3:29 下午 
'''

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.CatDogDataset import CatDogDataset
from neural_networks.linear.Linear import Liner
from neural_networks.activation.Relu import Relu
from neural_networks.activation.Sigmoid import Sigmoid
from neural_networks.loss.BCELoss import BCELoss

def get_data(image_path):
    train = CatDogDataset(image_path)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl

def get_model():

    models = [
        Liner(784,100),
        Relu(),
        Liner(100,1),
        Sigmoid()
        ]

    loss_fn = BCELoss()


    return models, loss_fn


def main():
    # image_path = "/Users/liujie/工作相关/data/DogCat/train_sample/*"
    image_path = "/Users/liujie/工作相关/data/DogCat/train/*"
    trn_dl = get_data(image_path)
    models, loss_fn = get_model()
    lrn_rate = 0.1
    epoch_num = 5

    losses = []
    for epoch in range(epoch_num):
        print("epoch {}:".format(epoch))

        epoch_losses, epoch_accuracies = [], []

        for ix, batch in enumerate(iter(trn_dl)):
            print("{} batch: ".format(ix))

            x, y = batch
            x, y = (x.numpy(),y.numpy()) if torch.is_tensor(y) else (x, y)

            gradients = []

            for model in models:
                gradient = model.grad(x)
                gradients.append(np.mean(gradient))
                x = model(x)

            batch_loss = loss_fn(x, y)

            epoch_losses.append(batch_loss)

            gradient = np.mean(loss_fn.grad(x, y))
            print("the batch gradient is {}: ".format(gradient))
            print("the batch loss is {}: ".format(batch_loss))

            if 0 not in gradients:
                for model,grad in zip(models[::-1], gradients[::-1]):

                    gradient *= grad
                    if isinstance(model,Liner):
                        model.weights -= lrn_rate * gradient


        epoch_loss = np.array(epoch_losses).mean()
        losses.append(epoch_loss)
    print(losses)



if __name__ == '__main__':
    main()