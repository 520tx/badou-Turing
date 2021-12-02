import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    # 初始化方法
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 定义损失函数loss
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]
        pass

    # 定义自动微分工具
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]
        pass

    # 定义训练方法
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad() # 所有梯度置零
                # forward --> 前向推理
                outputs = self.net(inputs)
                # backward + optimize --> 反向传播 + 优化
                loss = self.cost(outputs, labels) # 计算损失
                loss.backward() # 反向传播求梯度
                self.optimizer.step() # 更新权重参数
                # 统计损失
                running_loss += loss.item()
                if i % 100 == 0:
                    # 输出 [epoch, 一个 epoch 中训练完成的 batch 的占比], 一个batch的损失loss
                    print("[epoch %d, %.2f%%] loss: %.3f" % (
                        epoch + 1, (i + 1) * 100. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print("Finished Training")
        pass

    # 定义评估方法
    def evaluate(self, test_loader):
        print("Evaluating ...")
        correct = 0
        total = 0
        with torch.no_grad(): # 测试和预测时没有梯度
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
        pass

def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0], [1])])
    # 训练数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    # 测试数据集
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader
    pass

class MnistNet(torch.nn.Module):
    # 将需要训练参数的层写在 init 函数中
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)
        pass

    # 参数不需要训练的层在 forward 方法中实现
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
        pass

if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)