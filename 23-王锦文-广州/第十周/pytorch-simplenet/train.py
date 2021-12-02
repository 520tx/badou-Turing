#coding=utf-8
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import os
from read_data import SimpleDataSet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''
该文件使用pytorch1.2实现简单神经网络，在windows7上执行，程序测试结束后无法退出，但是在ubuntu上运行能正常退出
，通过查资料，认为这是pytorch1.2在windows7上的一个bug。
'''
class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(SimpleNet, self).__init__()
      
        self.layer1 = nn.Sequential(
                        nn.Linear(in_dim, n_hidden_1),
                        nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(n_hidden_1, out_dim)
        self.softmax=nn.Softmax(dim=1)
        #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x,btrain=True):
        out = self.layer1(x)
        out = self.layer2(out)
        if btrain==False:
            out=self.softmax(out)
        return out

def train(num_epochs,batch_size=4,train_path=None,val_path=None):
    #训练集
    train_dataset=SimpleDataSet(datapath=train_path)
    #测试集
    test_dataset=SimpleDataSet(datapath=val_path)
    #创建dataloader
    train_dataloders=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,#是否将数据保存在pin_memory ,这样转GPU会快一点,这里数据较少，用False
        drop_last=True
        )
    val_dataloders=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,#
        drop_last=False
        )
    #创建模型
    model=SimpleNet(784,200,10)
   # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    model.to(device=device)
    #定义loss
    criterion=nn.CrossEntropyLoss()
    #定义优化器
    optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.00004)
    #学习率衰减策略
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)#每间隔5个epoch衰减一次
    
    for epoch in range(num_epochs):
        runing_loss=0.#平均loss
        for i,(inputs,labels) in enumerate(train_dataloders):
            inputs=inputs.to(device)

            labels=labels.long().to(device)
            #forward
            outputs=model(inputs)

            loss=criterion(outputs,labels)
            loss=loss.mean()
            loss.backward()
            #梯度更新
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step(epoch)
            runing_loss+=loss
            batch_loss=runing_loss.item()/((i+1)*batch_size)
            print("epoch={},batchloss={}".format(epoch,batch_loss))
    
    #训练完成后测试
    total_num=0.0
    corrects_num = 0.0
    model=model.eval()
    for i,(inputs,labels) in enumerate(val_dataloders):
        inputs=inputs.to(device)
        labels=labels.long().to(device)
        #forward
        outputs=model(inputs,btrain=False).detach().cpu()
        _,preds=torch.max(outputs,1)
        labels=labels.cpu()
        total_num+=labels.size(0)
        corrects_num += float(torch.sum(preds == labels))
    print("test dataset acc:",corrects_num/total_num)

if __name__=="__main__":
    train(num_epochs=10,batch_size=4,train_path="../dataset/mnist_train.csv",val_path="../dataset/mnist_test.csv")



            



    
    


