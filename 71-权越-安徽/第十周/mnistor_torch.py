import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import numpy as np


class Model:
    def __init__(self,net,cost,optimist):
        self.net=net
        self.cost=self.create_cost(cost)
        self.optimizer=self.create_optimizer(optimist)

    def create_cost(self,cost):
        support_cost={
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE':nn.MSELoss()
        }
        return support_cost[cost]

    # 获得梯度优化方法，其中SGD with momentum 已经在optim.SGD中的参数momentum中实现，
    def create_optimizer(self,optimist,**rests):
        support_optim={
            'SGD':optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM':optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            running_loss=0.0

            for i ,data in enumerate(train_loader):

                inputs,labels=data
                self.optimizer.zero_grad()

                outputs=self.net(inputs)
                loss=self.cost(outputs,labels)
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()

                if i%100==0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss=0.0
        print('Finished Training')

    def evaluate(self,test_loader):
        print('Evalutaing....')
        correct=0
        total=0
        with torch.no_grad():
            for data in test_loader:
                images,labels=data
                outputs=self.net(images)
                predicted=torch.argmax(outputs,1)

                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def load_data():
    # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
    # 标准化处理 转换为标准正太分布，均值为0，方差为1
    trans=transforms.Compose(
        [ transforms.ToTensor(),
                           transforms.Normalize((0.1037,), (0.3081,))
         ]
    )
    # 拉取训练数据
    trainset=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform =trans)
    # 按照批次大小获得数据
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True)

    # 拉取测试数据
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform =trans)
    # 按照批次大小获得数据
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    return trainloader,testloader




# 定义网络结构
class MnistNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(28*28,512)
        self.fc2=torch.nn.Linear(512,512)
        self.fc3=torch.nn.Linear(512,10)

    def forward(self,x):
        x=x.reshape(-1,28*28)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x),dim=1)
        return x

def predict( imgPath):
    img = Image.open(imgPath)
    img=img.convert('RGB')
    img = img.resize((28, 28),Image.NEAREST)
    trans=transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
         ]
    )
    img=trans(img)
    img = img.view(1, 1, 28, 28)


    model=torch.load("model.pth")

    # model.eval()
    with torch.no_grad():
        output = model(img)

    _, prediction = torch.max(output, 1)
    #将预测结果从tensor转为array，并抽取结果
    prediction = prediction.numpy()[0]
    print ("预测结果:",prediction)
    return prediction

if __name__=='__main__':
    # 训练模型，保存模型
    # net=MnistNet()
    # model=Model(net,'CROSS_ENTROPY', 'RMSP')
    # train_loader,test_loader=load_data()
    #
    # model.train(train_loader)
    # model.evaluate(test_loader)
    # torch.save(net, 'model.pth')

    # # 加载模型预测结果
    predict("test2.png")
    predict("test.png")
    predict("test3.png")