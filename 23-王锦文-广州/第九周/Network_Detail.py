#coding=utf-8
from sys import maxsize
import numpy as np
'''
简单实现BP网络,为方便实现扩展层，分别对不同层封装
'''
class FCLayer:
    '''
    详细实现全连接层
    '''
    def __init__(self,inputchannels,outputchannels):
        '''
        inputchannels:输入的节点个数
        outputchannels：输出节点个数
        '''
        self.inputchannels=inputchannels
        self.outputchannels=outputchannels
        self.bias=np.zeros(self.outputchannels)
        self.W= np.random.normal(0.0, pow(self.inputchannels, -0.5), (self.inputchannels,self.outputchannels))
        self.dw=np.zeros((self.inputchannels,self.outputchannels))#参数的梯度值 w=w-lr*self.dw
        self.dbias=np.zeros((self.outputchannels,1))#参数的偏置梯度值
        self.inputs=None;#输入的信息

    def forward(self,x):
        '''
        网络的前向传播
        '''
        #1.全连接的正向传播，其实就是权重乘以输入求和+bias
        self.inputs=x
        output=np.dot(x,self.W)+self.bias
        return output

    def backward(self,diff):
        '''
        网络反向传播，diff：累计的梯度误差
        '''
        #全连接的反向传播，这个也比较简单，
        self.dw=np.dot(self.inputs.T,diff)#这里是根据公式得,误差对当前的权重偏导=上一层的输出*累计误差
       # self.dbias=diff#对偏置求导数，只剩下diff
        self.dbias=np.sum(diff, axis=0)
        diff=np.dot(diff,self.W.T)#当前层的输入误差
        return diff

    def step(self,lr):
        '''
        更新本层参数
        lr：学习率
        '''

        self.W=self.W-lr*self.dw
        self.bias=self.bias-lr*self.dbias

class  Sigmoid:
    '''
    Sigmoid的正向和反向传播实现
    '''
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, diff):
        dx = diff * self.out * (1 - self.out)
        return dx
    def step(self,lr):
        pass


class SimpleNetwork:
    def __init__(self,lr=0.01):
        self.layers=[]#存储网络层列表，为了方便哪些层需要更新参数，这里的列表里存储的数据是字典，key值是对应层的名称，value是对应的层
        self.lr=lr

    def appendlayer(self,layer):
        '''
        添加网络层
        '''
        self.layers.append(layer)
    def forward(self,x):
        '''
        前向网络
        '''

        for layer in self.layers:
            for k,l in layer.items():
                x=l.forward(x)
        return x
    def backward(self,diff):
        '''
        反向传播实现
        '''
        #误差反向传播
        for i in reversed(range(len(self.layers))):#这里我们从最后一层开始进行反向传播
            for k,l in self.layers[i].items():
                diff=l.backward(diff)
    
    def step(self,lr):
        '''
        对应层的参数更新
        '''
        for layer in self.layers:
            for k,l in layer.items():
                #if 'fc' in k:#是fc层才需要更新参数
                l.step(lr)

    def load_data(self,path):
        '''
        读取数据集
        '''
        with open(path,'r') as f:
            self.training_data_list = f.readlines()
        all_inputs=[]
        all_targets=[]
        for line in self.training_data_list:
            values=line.split(",")
            inputs = (np.asfarray(values[1:]))/255.0 * 0.99 + 0.01
            targets=int(values[0])#标签
            all_inputs.append(inputs)
            all_targets.append(targets)
        all_inputs=np.array(all_inputs)
        all_targets=np.array(all_targets)
        all_targets=all_targets.reshape(all_inputs.shape[0],1)
        self.total_inputs=np.concatenate((all_inputs,all_targets),axis=1)
        return self.total_inputs
        
        
    def shuffle_data(self):
        '''
        打乱数据顺序
        '''
        np.random.shuffle(self.total_inputs)

    def train(self,epochs=10,batch_size=4,datapath=None):
        '''
        对模型进行训练
        '''
        #对数据进行读取
        self.load_data(datapath)
        #获取一个epoch总的batch数量
        max_batch_len=int(self.total_inputs.shape[0]/batch_size)
        #训练
        step_index=0
        print("max_batch_len:",max_batch_len)
        for epoch in range(epochs):
            #先进行一次数据随机打乱
            self.shuffle_data()
            #学习率控制，这里模拟一下衰减
            if epoch>0.7*epochs:
                step_index=1
            lr=self.lr*(0.1**step_index)
            for bs in range(max_batch_len):
                #获取一个batch的数据
                images=self.total_inputs[bs*batch_size:(bs+1)*batch_size,:-1]
                labels=self.total_inputs[bs*batch_size:(bs+1)*batch_size,-1].astype(np.int32)
                #前向
                output=self.forward(images)
                #将标签变为one-hot
                targets_onehot=np.zeros_like(output)+0.01
                targets_onehot[np.arange(output.shape[0]),labels]=0.99
                #2.计算误差(平均误差)
                loss = np.mean(np.sum(np.square(output-targets_onehot), axis=-1))
                #误差梯度
                diff = output - targets_onehot
              #   #反向传播
                self.backward(diff)
                #参数更新
                self.step(lr)
                print("epoch={},lr={:.6f},loss={}".format(epoch, lr, loss))
    def save_param(self):
        '''
        每个epoch训练结束后，还可以把权重和偏置保存一下，方便训练完后加载参数进行预测，这里先不保存了
        '''
        pass

if __name__=='__main__':
    #先随意构建一个网络
    model=SimpleNetwork(lr=0.1)
    model.appendlayer({'fc1':FCLayer(784,256)})
    model.appendlayer({'activate1':Sigmoid()})
    model.appendlayer({'fc2':FCLayer(256,10)})
    model.appendlayer({'activate2': Sigmoid()})
    model.train(datapath="./dataset/mnist_train.csv",epochs=8,batch_size=4)
    #训练完成后对数据进行测试工作
    test_data_list=model.load_data("./dataset/mnist_test.csv")
    all_predlabel=[]
    for i in range(test_data_list.shape[0]):
        input,label=test_data_list[i][:-1],test_data_list[i][-1]
        #预测
        prob=model.forward(input)
        #找最大值对应索引
        predlabel=np.argmax(prob)
        all_predlabel.append(predlabel)
    gt=test_data_list[:,-1]#真实标签
    acc=sum((all_predlabel==gt))/len(gt)
    print("testset accuracy :",acc)










    


    




       

        


