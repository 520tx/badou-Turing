from torch import nn
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import time


class Network(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_channel, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, out_channel)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_dataloader(batch_size, data_dir="./minist"):
    transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ])
    trainset = datasets.MNIST(root=data_dir, train=True, transform=transfrom, download=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    valset = datasets.MNIST(root=data_dir, train=False, transform=transfrom, download=True)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader


def train(net, train_dataloader, val_dataloader, criterion, optimizer, epochs, batch_size, device):
    for epoch in range(epochs):
        total_loss = 0.0
        n_bacth = 0
        t1 = time.time()
        for batch in train_dataloader:
            optimizer.zero_grad()
            prediction = net(batch[0].reshape(batch_size, -1))
            loss = criterion(prediction, batch[1])
            loss.backward()
            optimizer.step()
            total_loss += loss
            n_bacth += 1
        print("[epoch {}/{}], train loss: {:.4f}, time: {:.4f} sec".format(
            epoch, epochs, total_loss/n_bacth, time.time()-t1))

        # 测试
        if epoch % 10 == 0:
            accuracy = 0
            n_iter = 0
            for batch in val_dataloader:
                prediction = net(batch[0].reshape(batch_size, -1))
                # print(torch.argmax(prediction, dim=1), batch[1])
                accuracy += torch.sum(torch.argmax(prediction, dim=1) == batch[1])/prediction.shape[0]
                n_iter += 1
            print("[epoch {}/{}] Test accuracy: {:.4f}".format(epoch, epochs, accuracy/n_iter))


if __name__ == "__main__":
    size = 28*28
    nClass = 10
    batch_size = 8
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network(size, nClass)
    train_dataloader, val_dataloader = get_dataloader(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-5)
    train(net, train_dataloader, val_dataloader, criterion, optimizer, epochs, batch_size, device)

